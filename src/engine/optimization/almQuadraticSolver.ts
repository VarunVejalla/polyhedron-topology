import { zeroQuadratic } from "./quadratic";
import type {
  IndexedQuadraticConstraint,
  OptimizationModel,
  QuadraticConstraint,
} from "./types";

type AlmQuadraticParams = {
  rho: number;
  proximalWeight: number;
  activeSetEps: number;
};

type AlmQuadraticDiagnostics = {
  primalResidual: number;
  stepResidual: number;
  activeIneq: number;
};

type IndexedConstraintLists = {
  equalities: IndexedQuadraticConstraint[];
  inequalities: IndexedQuadraticConstraint[];
};

function matVec(A: ReadonlyArray<ReadonlyArray<number>>, x: ReadonlyArray<number>): number[] {
  const out = new Array<number>(x.length).fill(0);
  for (let i = 0; i < A.length; i++) {
    let s = 0;
    const row = A[i] ?? [];
    for (let j = 0; j < x.length; j++) s += (row[j] ?? 0) * x[j];
    out[i] = s;
  }
  return out;
}

function vecNorm2(x: ReadonlyArray<number>): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return s;
}

function clone2D(A: ReadonlyArray<ReadonlyArray<number>>): number[][] {
  return A.map((r) => [...r]);
}

function solveLinearSystem(Ain: ReadonlyArray<ReadonlyArray<number>>, bin: ReadonlyArray<number>): number[] | null {
  const n = bin.length;
  const A = clone2D(Ain);
  const b = [...bin];
  for (let k = 0; k < n; k++) {
    let pivot = k;
    let best = Math.abs(A[k]?.[k] ?? 0);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(A[i]?.[k] ?? 0);
      if (v > best) {
        best = v;
        pivot = i;
      }
    }
    if (!(best > 1e-14)) return null;
    if (pivot !== k) {
      const row = A[k];
      A[k] = A[pivot];
      A[pivot] = row;
      const t = b[k];
      b[k] = b[pivot];
      b[pivot] = t;
    }
    const Akk = A[k][k];
    for (let i = k + 1; i < n; i++) {
      const f = (A[i][k] ?? 0) / Akk;
      if (f === 0) continue;
      A[i][k] = 0;
      for (let j = k + 1; j < n; j++) A[i][j] -= f * (A[k][j] ?? 0);
      b[i] -= f * b[k];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let rhs = b[i];
    for (let j = i + 1; j < n; j++) rhs -= (A[i][j] ?? 0) * x[j];
    const aii = A[i][i];
    if (!(Math.abs(aii) > 1e-14)) return null;
    x[i] = rhs / aii;
  }
  return x;
}

function asIndexedConstraint(constraint: QuadraticConstraint, dim: number): IndexedQuadraticConstraint {
  return {
    id: constraint.id,
    sense: constraint.sense,
    form: {
      indices: Array.from({ length: dim }, (_, i) => i),
      A: constraint.form.A,
      b: constraint.form.b,
      c: constraint.form.c,
    },
  };
}

function collectConstraints(model: OptimizationModel, dim: number): IndexedConstraintLists {
  const indexed = model.indexedQuadraticConstraints ?? { equalities: [], inequalities: [] };
  const equalities = [...indexed.equalities];
  const inequalities = [...indexed.inequalities];
  for (let i = 0; i < model.quadraticConstraints.equalities.length; i++) {
    equalities.push(asIndexedConstraint(model.quadraticConstraints.equalities[i], dim));
  }
  for (let i = 0; i < model.quadraticConstraints.inequalities.length; i++) {
    inequalities.push(asIndexedConstraint(model.quadraticConstraints.inequalities[i], dim));
  }
  return { equalities, inequalities };
}

function evalIndexed(
  c: IndexedQuadraticConstraint,
  x: ReadonlyArray<number>
): { value: number; grad: number[]; indices: number[] } {
  const indices = c.form.indices;
  const m = indices.length;
  const y = new Array<number>(m);
  for (let i = 0; i < m; i++) y[i] = x[indices[i]];
  const grad = new Array<number>(m).fill(0);
  let value = c.form.c;
  for (let i = 0; i < m; i++) value += (c.form.b[i] ?? 0) * y[i];
  for (let i = 0; i < m; i++) {
    let ay = 0;
    const row = c.form.A[i] ?? [];
    for (let j = 0; j < m; j++) ay += (row[j] ?? 0) * y[j];
    grad[i] = ay + (c.form.b[i] ?? 0);
    value += 0.5 * y[i] * ay;
  }
  return { value, grad, indices };
}

function accumulateQuadraticSpace(
  H: number[][],
  g: number[],
  evalAtXk: { value: number; grad: number[]; indices: number[] },
  lambda: number,
  rho: number
): void {
  // Gauss-Newton ALM model: linearize h(x) first, then square.
  // This keeps penalty curvature PSD via rho * grad * grad^T and avoids
  // indefinite second-order terms from Hessian(h).
  const coeff = lambda + rho * evalAtXk.value;
  const m = evalAtXk.indices.length;
  for (let i = 0; i < m; i++) {
    const gi = evalAtXk.grad[i];
    const ii = evalAtXk.indices[i];
    g[ii] += coeff * gi;
    for (let j = 0; j < m; j++) {
      const jj = evalAtXk.indices[j];
      H[ii][jj] += rho * gi * evalAtXk.grad[j];
    }
  }
}

export class AlmQuadraticSolver {
  private readonly model: OptimizationModel;
  private readonly dim: number;
  private readonly constraints: IndexedConstraintLists;
  private x: number[];
  private readonly lambdaEq: number[];
  private readonly lambdaIneq: number[];
  private readonly rhoEq: number[];
  private readonly rhoIneq: number[];
  private params: AlmQuadraticParams;
  private lastDiag: AlmQuadraticDiagnostics = {
    primalResidual: 0,
    stepResidual: 0,
    activeIneq: 0,
  };

  constructor(args: {
    model: OptimizationModel;
    initialX: number[];
    rho: number;
    proximalWeight?: number;
    activeSetEps?: number;
  }) {
    this.model = args.model;
    this.dim = args.initialX.length;
    this.constraints = collectConstraints(this.model, this.dim);
    this.x = [...args.initialX];
    this.lambdaEq = new Array<number>(this.constraints.equalities.length).fill(0);
    this.lambdaIneq = new Array<number>(this.constraints.inequalities.length).fill(0);
    this.rhoEq = new Array<number>(this.constraints.equalities.length).fill(Math.max(1e-8, args.rho));
    this.rhoIneq = new Array<number>(this.constraints.inequalities.length).fill(Math.max(1e-8, args.rho));
    this.params = {
      rho: Math.max(1e-8, args.rho),
      proximalWeight: Math.max(0, args.proximalWeight ?? 1e-6),
      activeSetEps: Math.max(0, args.activeSetEps ?? 1e-10),
    };
  }

  resetState(nextX: number[]): void {
    this.x = [...nextX];
    this.lambdaEq.fill(0);
    this.lambdaIneq.fill(0);
    this.lastDiag = {
      primalResidual: 0,
      stepResidual: 0,
      activeIneq: 0,
    };
  }

  setParams(next: Partial<AlmQuadraticParams>): void {
    this.params = { ...this.params, ...next };
    const rho = Math.max(1e-8, this.params.rho);
    for (let i = 0; i < this.rhoEq.length; i++) this.rhoEq[i] = rho;
    for (let i = 0; i < this.rhoIneq.length; i++) this.rhoIneq[i] = rho;
    this.params.rho = rho;
    this.params.proximalWeight = Math.max(0, this.params.proximalWeight);
    this.params.activeSetEps = Math.max(0, this.params.activeSetEps);
  }

  getStateRef(): ReadonlyArray<number> {
    return this.x;
  }

  snapshotState(): number[] {
    return [...this.x];
  }

  diagnostics(): AlmQuadraticDiagnostics {
    return this.lastDiag;
  }

  step(iterations: number): void {
    for (let it = 0; it < iterations; it++) {
      const xk = [...this.x];
      const baseMetric = this.model.localQuadraticMetric?.(xk) ?? zeroQuadratic(this.dim);
      const baseReg = this.model.localQuadraticRegularizer?.(xk) ?? zeroQuadratic(this.dim);
      const H = clone2D(baseMetric.A);
      const b = matVec(baseMetric.A, xk).map((v, i) => v + (baseMetric.b[i] ?? 0));
      const regScale = 1;
      for (let i = 0; i < this.dim; i++) {
        const rowReg = baseReg.A[i] ?? [];
        for (let j = 0; j < this.dim; j++) H[i][j] += regScale * (rowReg[j] ?? 0);
      }
      const regGrad = matVec(baseReg.A, xk);
      for (let i = 0; i < this.dim; i++) b[i] += regScale * (regGrad[i] + (baseReg.b[i] ?? 0));

      let activeIneq = 0;
      for (let i = 0; i < this.constraints.equalities.length; i++) {
        const evalAt = evalIndexed(this.constraints.equalities[i], xk);
        accumulateQuadraticSpace(H, b, evalAt, this.lambdaEq[i], this.rhoEq[i]);
      }
      for (let i = 0; i < this.constraints.inequalities.length; i++) {
        const evalAt = evalIndexed(this.constraints.inequalities[i], xk);
        const rho = this.rhoIneq[i];
        if (evalAt.value + this.lambdaIneq[i] / rho <= this.params.activeSetEps) continue;
        activeIneq++;
        accumulateQuadraticSpace(H, b, evalAt, this.lambdaIneq[i], rho);
      }

      for (let i = 0; i < this.dim; i++) H[i][i] += this.params.proximalWeight;
      const rhs = b.map((v) => -v);
      let d = solveLinearSystem(H, rhs);
      if (!d) {
        for (let i = 0; i < this.dim; i++) H[i][i] += 1e-4;
        d = solveLinearSystem(H, rhs);
      }
      if (!d) break;

      for (let i = 0; i < this.dim; i++) this.x[i] = xk[i] + d[i];
      let primal2 = 0;
      for (let i = 0; i < this.constraints.equalities.length; i++) {
        const v = evalIndexed(this.constraints.equalities[i], this.x).value;
        this.lambdaEq[i] += this.rhoEq[i] * v;
        primal2 += v * v;
      }
      for (let i = 0; i < this.constraints.inequalities.length; i++) {
        const g = evalIndexed(this.constraints.inequalities[i], this.x).value;
        this.lambdaIneq[i] = Math.max(0, this.lambdaIneq[i] + this.rhoIneq[i] * g);
        const gp = Math.max(0, g);
        primal2 += gp * gp;
      }

      this.lastDiag = {
        primalResidual: Math.sqrt(primal2),
        stepResidual: Math.sqrt(vecNorm2(d)),
        activeIneq,
      };
    }
  }
}
