import { projectQcqp1 } from "../qcqp1Projector";
import type {
  ConstraintSense,
  IndexedQuadraticForm,
  OptimizationModel,
  OptimizerKernel,
  QuadraticForm,
} from "../types";

export type ConsensusConstraint = {
  id: string;
  sense: ConstraintSense;
  form: IndexedQuadraticForm;
};

export type ConsensusQcqpModel = {
  dim: number;
  objective: QuadraticForm;
  constraints: ConsensusConstraint[];
  sourceModel?: OptimizationModel;
};

export type ConsensusQcqpState = {
  x: number[];
};

export type ConsensusQcqpParams = {
  rho: number;
  proximalWeight: number;
  linearSolveShift: number;
  qcqpTol: number;
  qcqpMaxNewtonIters: number;
};

type LocalBlock = {
  id: string;
  sense: ConstraintSense;
  indices: number[];
  form: IndexedQuadraticForm;
  z: number[];
  u: number[];
};

export type ConsensusQcqpMemory = {
  locals: LocalBlock[];
  zById: Record<string, number[]>;
  uById: Record<string, number[]>;
  diagCounts: number[];
  primalResidual: number;
  dualResidual: number;
};

function dot(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function symmetrize(A: number[][]): number[][] {
  const n = A.length;
  const out = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      out[i][j] = 0.5 * ((A[i]?.[j] ?? 0) + (A[j]?.[i] ?? 0));
    }
  }
  return out;
}

function cholesky(M: ReadonlyArray<ReadonlyArray<number>>): number[][] | null {
  const n = M.length;
  const L = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = M[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 1e-14 || !Number.isFinite(s)) return null;
        L[i][j] = Math.sqrt(s);
      } else {
        if (Math.abs(L[j][j]) < 1e-14) return null;
        L[i][j] = s / L[j][j];
      }
    }
  }
  return L;
}

function solveWithCholesky(L: ReadonlyArray<ReadonlyArray<number>>, b: ReadonlyArray<number>): number[] {
  const n = b.length;
  const y = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
    y[i] = s / L[i][i];
  }
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

function solveX(
  objective: Readonly<QuadraticForm>,
  rhs: ReadonlyArray<number>,
  diagCounts: ReadonlyArray<number>,
  rho: number,
  prox: number,
  baseShift: number
): number[] {
  const A = symmetrize(objective.A);
  const n = rhs.length;
  let shift = Math.max(0, baseShift);
  for (let tries = 0; tries < 8; tries++) {
    const M = Array.from({ length: n }, (_, i) => {
      const row = new Array<number>(n);
      for (let j = 0; j < n; j++) row[j] = A[i]?.[j] ?? 0;
      row[i] += rho * (diagCounts[i] ?? 0) + prox + shift;
      return row;
    });
    const L = cholesky(M);
    if (L) return solveWithCholesky(L, rhs);
    shift = shift === 0 ? 1e-8 : shift * 10;
  }

  // Fallback: damped diagonal solve.
  const x = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    const d = (A[i]?.[i] ?? 0) + rho * (diagCounts[i] ?? 0) + prox + shift + 1e-6;
    x[i] = rhs[i] / d;
  }
  return x;
}

function cloneArrayMap(src?: Readonly<Record<string, number[]>>): Record<string, number[]> {
  const out: Record<string, number[]> = {};
  for (const k in src) out[k] = src[k].slice();
  return out;
}

function buildLocals(
  model: ConsensusQcqpModel,
  state: ConsensusQcqpState,
  resume?: ConsensusQcqpMemory
): { locals: LocalBlock[]; diagCounts: number[] } {
  const zById = resume?.zById ?? {};
  const uById = resume?.uById ?? {};
  const locals: LocalBlock[] = new Array(model.constraints.length);
  const diagCounts = new Array<number>(model.dim).fill(0);

  for (let i = 0; i < model.constraints.length; i++) {
    const c = model.constraints[i];
    const indices = c.form.indices.slice();
    const zPrev = zById[c.id];
    const uPrev = uById[c.id];
    const z = indices.map((idx, j) => zPrev && zPrev.length === indices.length ? zPrev[j] : state.x[idx]);
    const u = indices.map((_, j) => (uPrev && uPrev.length === indices.length ? uPrev[j] : 0));
    for (let j = 0; j < indices.length; j++) diagCounts[indices[j]] += 1;
    locals[i] = { id: c.id, sense: c.sense, indices, form: c.form, z, u };
  }
  return { locals, diagCounts };
}

export const consensusQcqpKernel: OptimizerKernel<
  ConsensusQcqpState,
  ConsensusQcqpParams,
  ConsensusQcqpMemory,
  ConsensusQcqpModel
> = {
  initialize: ({ model, state, resume }) => {
    if (state.x.length !== model.dim) state.x = state.x.slice(0, model.dim);
    while (state.x.length < model.dim) state.x.push(0);
    const built = buildLocals(model, state, resume);
    return {
      locals: built.locals,
      diagCounts: built.diagCounts,
      zById: cloneArrayMap(resume?.zById),
      uById: cloneArrayMap(resume?.uById),
      primalResidual: resume?.primalResidual ?? 0,
      dualResidual: resume?.dualResidual ?? 0,
    };
  },

  step: ({ model, state, params, memory, iterations }) => {
    if (iterations <= 0) return;
    const rho = Math.max(1e-8, params.rho);
    const prox = Math.max(0, params.proximalWeight);
    const shift = Math.max(0, params.linearSolveShift);

    for (let it = 0; it < iterations; it++) {
      const accum = new Array<number>(model.dim).fill(0);
      for (let i = 0; i < memory.locals.length; i++) {
        const block = memory.locals[i];
        for (let j = 0; j < block.indices.length; j++) {
          accum[block.indices[j]] += (block.z[j] ?? 0) - (block.u[j] ?? 0);
        }
      }

      const rhs = new Array<number>(model.dim);
      for (let k = 0; k < model.dim; k++) rhs[k] = -((model.objective.b[k] ?? 0)) + rho * accum[k];
      const xNew = solveX(model.objective, rhs, memory.diagCounts, rho, prox, shift);
      state.x = xNew;

      let primalSq = 0;
      let dualSq = 0;
      for (let i = 0; i < memory.locals.length; i++) {
        const block = memory.locals[i];
        const v = block.indices.map((idx, j) => state.x[idx] + block.u[j]);
        const zOld = block.z.slice();
        const proj = projectQcqp1(v, block.form, block.sense, {
          tol: params.qcqpTol,
          maxNewtonIters: params.qcqpMaxNewtonIters,
        });
        block.z = proj.z;

        for (let j = 0; j < block.indices.length; j++) {
          const r = state.x[block.indices[j]] - block.z[j];
          block.u[j] += r;
          primalSq += r * r;
          const dz = block.z[j] - zOld[j];
          dualSq += dz * dz;
        }

        memory.zById[block.id] = block.z.slice();
        memory.uById[block.id] = block.u.slice();
      }
      memory.primalResidual = Math.sqrt(primalSq);
      memory.dualResidual = rho * Math.sqrt(dualSq);

      const smallStep = dot(state.x, state.x) > 0
        ? memory.primalResidual / Math.max(1e-12, Math.sqrt(dot(state.x, state.x))) < 1e-9
        : memory.primalResidual < 1e-9;
      if (smallStep && memory.dualResidual < 1e-9) break;
    }
  },
};
