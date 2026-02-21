import { zeroQuadratic } from "./quadratic";
import { projectQcqp1Paper, type Qcqp1Constraint } from "./qcqp1Paper";
import type { IndexedQuadraticConstraint, OptimizationModel, QuadraticConstraint } from "./types";
import { DEFAULT_DAMPING, MIN_RHO, SOLVER_FALLBACK_DIAG, SOLVER_PIVOT_EPS } from "../math/constants";

type ConsensusQcqpParams = {
  rho: number;
  damping: number;
};

type ConstraintBlock = {
  id: string;
  sense: "eq" | "le";
  indices: number[];
  z: number[];
  zPrev: number[];
  u: number[];
  qcqp1: Qcqp1Constraint;
};

type ConsensusQcqpDiagnostics = {
  primalResidual: number;
  dualResidual: number;
  maxConstraintViolation: number;
};

function dot(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  let out = 0;
  for (let i = 0; i < a.length; i++) out += a[i] * b[i];
  return out;
}

function clone2D(A: ReadonlyArray<ReadonlyArray<number>>): number[][] {
  return A.map((r) => [...r]);
}

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
    if (!(best > SOLVER_PIVOT_EPS)) return null;
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
    if (!(Math.abs(aii) > SOLVER_PIVOT_EPS)) return null;
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

function collectIndexedConstraints(model: OptimizationModel, dim: number): IndexedQuadraticConstraint[] {
  const indexed = model.indexedQuadraticConstraints ?? { equalities: [], inequalities: [] };
  const out: IndexedQuadraticConstraint[] = [...indexed.equalities, ...indexed.inequalities];
  for (let i = 0; i < model.quadraticConstraints.equalities.length; i++) {
    out.push(asIndexedConstraint(model.quadraticConstraints.equalities[i], dim));
  }
  for (let i = 0; i < model.quadraticConstraints.inequalities.length; i++) {
    out.push(asIndexedConstraint(model.quadraticConstraints.inequalities[i], dim));
  }
  return out;
}

function extractByIndices(x: ReadonlyArray<number>, indices: ReadonlyArray<number>): number[] {
  const out = new Array<number>(indices.length);
  for (let i = 0; i < indices.length; i++) out[i] = x[indices[i]];
  return out;
}

function addScaledAtIndices(dst: number[], indices: ReadonlyArray<number>, src: ReadonlyArray<number>, scale: number): void {
  for (let i = 0; i < indices.length; i++) dst[indices[i]] += scale * src[i];
}

function combineObjective(model: OptimizationModel, x: ReadonlyArray<number>) {
  const m = model.localQuadraticMetric?.(x) ?? zeroQuadratic(x.length);
  const r = model.localQuadraticRegularizer?.(x) ?? zeroQuadratic(x.length);
  const A = clone2D(m.A);
  const b = [...m.b];
  let c = m.c;
  for (let i = 0; i < x.length; i++) {
    const row = r.A[i] ?? [];
    for (let j = 0; j < x.length; j++) A[i][j] += row[j] ?? 0;
    b[i] += r.b[i] ?? 0;
  }
  c += r.c;
  return { A, b, c };
}

function localToPaper(c: IndexedQuadraticConstraint): Qcqp1Constraint {
  // local form: 1/2 y^T A y + b^T y + c <= 0
  // paper form: y^T A y - 2 b^T y - c <= 0
  return {
    sense: c.sense,
    A: clone2D(c.form.A),
    b: c.form.b.map((v) => -v),
    c: -2 * c.form.c,
  };
}

function evalPaperConstraint(c: Qcqp1Constraint, z: ReadonlyArray<number>): number {
  const Az = matVec(c.A, z);
  return dot(z, Az) - 2 * dot(c.b, z) - c.c;
}

export class ConsensusQcqpSolver {
  private readonly model: OptimizationModel;
  private readonly dim: number;
  private params: ConsensusQcqpParams;
  private x: number[];
  private readonly blocks: ConstraintBlock[];
  private diag: ConsensusQcqpDiagnostics = {
    primalResidual: 0,
    dualResidual: 0,
    maxConstraintViolation: 0,
  };

  constructor(args: {
    model: OptimizationModel;
    initialX: number[];
    rho: number;
    damping?: number;
  }) {
    this.model = args.model;
    this.dim = args.initialX.length;
    this.params = {
      rho: Math.max(MIN_RHO, args.rho),
      damping: Math.max(0, args.damping ?? DEFAULT_DAMPING),
    };
    this.x = [...args.initialX];
    const constraints = collectIndexedConstraints(this.model, this.dim);
    this.blocks = constraints.map((c) => {
      const z0 = extractByIndices(this.x, c.form.indices);
      return {
        id: c.id,
        sense: c.sense,
        indices: [...c.form.indices],
        z: [...z0],
        zPrev: [...z0],
        u: new Array<number>(c.form.indices.length).fill(0),
        qcqp1: localToPaper(c),
      };
    });
  }

  setParams(next: Partial<ConsensusQcqpParams>): void {
    this.params = { ...this.params, ...next };
    this.params.rho = Math.max(MIN_RHO, this.params.rho);
    this.params.damping = Math.max(0, this.params.damping);
  }

  resetState(nextX: number[]): void {
    this.x = [...nextX];
    for (let i = 0; i < this.blocks.length; i++) {
      const blk = this.blocks[i];
      const z0 = extractByIndices(this.x, blk.indices);
      for (let j = 0; j < z0.length; j++) {
        blk.z[j] = z0[j];
        blk.zPrev[j] = z0[j];
        blk.u[j] = 0;
      }
    }
    this.diag = {
      primalResidual: 0,
      dualResidual: 0,
      maxConstraintViolation: 0,
    };
  }

  getStateRef(): ReadonlyArray<number> {
    return this.x;
  }

  snapshotState(): number[] {
    return [...this.x];
  }

  diagnostics(): ConsensusQcqpDiagnostics {
    return this.diag;
  }

  step(iterations: number): void {
    if (iterations <= 0 || this.blocks.length === 0) return;
    const rho = this.params.rho;
    for (let it = 0; it < iterations; it++) {
      const objective = combineObjective(this.model, this.x);
      const H = clone2D(objective.A);
      const rhs = objective.b.map((v) => -v);

      for (let bi = 0; bi < this.blocks.length; bi++) {
        const blk = this.blocks[bi];
        for (let li = 0; li < blk.indices.length; li++) {
          const gi = blk.indices[li];
          H[gi][gi] += rho;
        }
        addScaledAtIndices(rhs, blk.indices, blk.z, rho);
        addScaledAtIndices(rhs, blk.indices, blk.u, rho);
      }

      for (let i = 0; i < this.dim; i++) H[i][i] += this.params.damping;
      let xNew = solveLinearSystem(H, rhs);
      if (!xNew) {
        for (let i = 0; i < this.dim; i++) H[i][i] += SOLVER_FALLBACK_DIAG;
        xNew = solveLinearSystem(H, rhs);
      }
      if (!xNew) break;
      this.x = xNew;

      let primal2 = 0;
      let dual2 = 0;
      let maxViolation = 0;
      for (let bi = 0; bi < this.blocks.length; bi++) {
        const blk = this.blocks[bi];
        const xLocal = extractByIndices(this.x, blk.indices);
        const zeta = new Array<number>(xLocal.length);
        for (let i = 0; i < xLocal.length; i++) zeta[i] = xLocal[i] - blk.u[i];
        for (let i = 0; i < blk.z.length; i++) blk.zPrev[i] = blk.z[i];
        const proj = projectQcqp1Paper(zeta, blk.qcqp1);
        for (let i = 0; i < blk.z.length; i++) blk.z[i] = proj.z[i];

        for (let i = 0; i < blk.u.length; i++) {
          const r = blk.z[i] - xLocal[i];
          blk.u[i] += r;
          primal2 += r * r;
          const dz = blk.z[i] - blk.zPrev[i];
          dual2 += (rho * dz) * (rho * dz);
        }

        const v = evalPaperConstraint(blk.qcqp1, blk.z);
        const violation = blk.sense === "eq" ? Math.abs(v) : Math.max(0, v);
        if (violation > maxViolation) maxViolation = violation;
      }

      this.diag = {
        primalResidual: Math.sqrt(primal2),
        dualResidual: Math.sqrt(dual2),
        maxConstraintViolation: maxViolation,
      };
    }
  }
}
