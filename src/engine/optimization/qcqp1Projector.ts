import type { ConstraintSense, DenseMatrix, IndexedQuadraticForm } from "./types";

type QCQP1ProjectOptions = {
  tol?: number;
  maxNewtonIters?: number;
  maxBacktrackSteps?: number;
  maxStepAbs?: number;
  fallbackIters?: number;
};

type QCQP1ProjectResult = {
  z: number[];
  converged: boolean;
  iterations: number;
  qValue: number;
  primalViolation: number;
  mu: number;
  stationarityResidual: number;
  dualResidual: number;
  complementarityResidual: number;
  kktResidual: number;
  feasible: boolean;
};

function dot(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function matVec(A: Readonly<DenseMatrix>, x: ReadonlyArray<number>): number[] {
  const out = new Array<number>(x.length).fill(0);
  for (let i = 0; i < A.length; i++) {
    let s = 0;
    const row = A[i];
    for (let j = 0; j < x.length; j++) s += (row?.[j] ?? 0) * x[j];
    out[i] = s;
  }
  return out;
}

function evaluateForm(form: Readonly<IndexedQuadraticForm>, z: ReadonlyArray<number>): number {
  const Az = matVec(form.A, z);
  return 0.5 * dot(z, Az) + dot(form.b, z) + form.c;
}

function gradForm(A: Readonly<DenseMatrix>, b: ReadonlyArray<number>, z: ReadonlyArray<number>): number[] {
  const Az = matVec(A, z);
  return z.map((_, i) => (Az[i] ?? 0) + (b[i] ?? 0));
}

function norm2(v: ReadonlyArray<number>): number {
  return Math.sqrt(dot(v, v));
}

function estimateMu(
  zeta: ReadonlyArray<number>,
  z: ReadonlyArray<number>,
  gradQ: ReadonlyArray<number>,
  qVal: number,
  sense: ConstraintSense,
  tol: number
): number {
  if (sense === "le" && qVal <= tol) return 0;
  const denom = Math.max(1e-14, dot(gradQ, gradQ));
  const numer = -dot(z.map((zi, i) => zi - zeta[i]), gradQ);
  const mu = numer / denom;
  if (sense === "le") return Math.max(0, mu);
  return mu;
}

function buildKktDiagnostics(args: {
  zeta: ReadonlyArray<number>;
  z: ReadonlyArray<number>;
  A: Readonly<DenseMatrix>;
  b: ReadonlyArray<number>;
  c: number;
  sense: ConstraintSense;
  tol: number;
  muHint?: number;
}): Omit<QCQP1ProjectResult, "z" | "converged" | "iterations"> {
  const qValue = 0.5 * dot(args.z, matVec(args.A, args.z)) + dot(args.b, args.z) + args.c;
  const gradQ = gradForm(args.A, args.b, args.z);
  const mu = args.muHint ?? estimateMu(args.zeta, args.z, gradQ, qValue, args.sense, args.tol);

  const stationarity = args.z.map((zi, i) => zi - args.zeta[i] + mu * gradQ[i]);
  const stationarityResidual = norm2(stationarity);
  const primalViolation = args.sense === "eq" ? Math.abs(qValue) : Math.max(0, qValue);
  const dualResidual = args.sense === "le" ? Math.max(0, -mu) : 0;
  const complementarityResidual = args.sense === "le" ? Math.abs(mu * qValue) : 0;
  const kktResidual = Math.max(stationarityResidual, primalViolation, dualResidual, complementarityResidual);
  const feasible = primalViolation <= args.tol;

  return {
    qValue,
    primalViolation,
    mu,
    stationarityResidual,
    dualResidual,
    complementarityResidual,
    kktResidual,
    feasible,
  };
}

function symmetrize(A: Readonly<DenseMatrix>): DenseMatrix {
  const n = A.length;
  const out = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const aij = A[i]?.[j] ?? 0;
      const aji = A[j]?.[i] ?? 0;
      out[i][j] = 0.5 * (aij + aji);
    }
  }
  return out;
}

function solveLinear(A: Readonly<DenseMatrix>, b: ReadonlyArray<number>): number[] | null {
  const n = b.length;
  const M = Array.from({ length: n }, (_, i) => Array.from({ length: n + 1 }, (_, j) => (j < n ? (A[i]?.[j] ?? 0) : b[i])));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    let best = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(M[r][col]);
      if (v > best) {
        best = v;
        pivot = r;
      }
    }
    if (!Number.isFinite(best) || best < 1e-14) return null;
    if (pivot !== col) {
      const tmp = M[col];
      M[col] = M[pivot];
      M[pivot] = tmp;
    }
    const diag = M[col][col];
    for (let j = col; j <= n; j++) M[col][j] /= diag;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      if (f === 0) continue;
      for (let j = col; j <= n; j++) M[r][j] -= f * M[col][j];
    }
  }
  const x = new Array<number>(n);
  for (let i = 0; i < n; i++) x[i] = M[i][n];
  return x;
}

function evalAtMu(
  mu: number,
  zeta: ReadonlyArray<number>,
  A: Readonly<DenseMatrix>,
  b: ReadonlyArray<number>,
  c: number
): { z: number[]; phi: number; phiPrime: number } | null {
  const n = zeta.length;
  const M = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0) + mu * (A[i]?.[j] ?? 0))
  );
  const rhs = zeta.map((v, i) => v - mu * (b[i] ?? 0));
  const z = solveLinear(M, rhs);
  if (!z) return null;
  const Az = matVec(A, z);
  const v = z.map((_, i) => (Az[i] ?? 0) + (b[i] ?? 0));
  const w = solveLinear(M, v);
  if (!w) return null;
  const phi = 0.5 * dot(z, Az) + dot(b, z) + c;
  const phiPrime = -dot(v, w);
  if (!Number.isFinite(phi) || !Number.isFinite(phiPrime)) return null;
  return { z, phi, phiPrime };
}

function fallbackPenalty(
  zeta: ReadonlyArray<number>,
  A: Readonly<DenseMatrix>,
  b: ReadonlyArray<number>,
  c: number,
  maxIters: number,
  tol: number
): number[] {
  let z = zeta.slice();
  const step = 0.1;
  let kappa = 8;
  for (let it = 0; it < maxIters; it++) {
    const Az = matVec(A, z);
    const q = 0.5 * dot(z, Az) + dot(b, z) + c;
    if (Math.abs(q) <= tol) break;
    const gradQ = z.map((_, i) => (Az[i] ?? 0) + (b[i] ?? 0));
    const grad = z.map((zi, i) => (zi - zeta[i]) + kappa * q * gradQ[i]);
    z = z.map((zi, i) => zi - step * grad[i]);
    kappa = Math.min(1e6, kappa * 1.05);
  }
  return z;
}

export function projectQcqp1(
  zeta: ReadonlyArray<number>,
  form: Readonly<IndexedQuadraticForm>,
  sense: ConstraintSense,
  opts: QCQP1ProjectOptions = {}
): QCQP1ProjectResult {
  const tol = opts.tol ?? 1e-8;
  const maxNewtonIters = opts.maxNewtonIters ?? 40;
  const maxBacktrackSteps = opts.maxBacktrackSteps ?? 16;
  const maxStepAbs = opts.maxStepAbs ?? 50;
  const fallbackIters = opts.fallbackIters ?? 60;

  const A = symmetrize(form.A);
  const b = form.b.slice();
  const c = form.c;

  const q0 = evaluateForm({ ...form, A }, zeta);
  if (sense === "le" && q0 <= tol) {
    return {
      z: zeta.slice(),
      converged: true,
      iterations: 0,
      ...buildKktDiagnostics({
        zeta,
        z: zeta.slice(),
        A,
        b,
        c,
        sense,
        tol,
        muHint: 0,
      }),
    };
  }

  let mu = 0;
  let best: { z: number[]; absPhi: number } | null = null;

  for (let it = 0; it < maxNewtonIters; it++) {
    const cur = evalAtMu(mu, zeta, A, b, c);
    if (!cur) break;
    const absPhi = Math.abs(cur.phi);
    if (!best || absPhi < best.absPhi) best = { z: cur.z, absPhi };
    if (absPhi <= tol) {
      return {
        z: cur.z,
        converged: true,
        iterations: it + 1,
        ...buildKktDiagnostics({
          zeta,
          z: cur.z,
          A,
          b,
          c,
          sense,
          tol,
          muHint: mu,
        }),
      };
    }

    const denom = Math.abs(cur.phiPrime) < 1e-14 ? (cur.phiPrime >= 0 ? 1e-14 : -1e-14) : cur.phiPrime;
    let step = -cur.phi / denom;
    if (step > maxStepAbs) step = maxStepAbs;
    if (step < -maxStepAbs) step = -maxStepAbs;
    if (!Number.isFinite(step)) break;

    let accepted = false;
    let alpha = 1;
    for (let bt = 0; bt < maxBacktrackSteps; bt++) {
      const cand = evalAtMu(mu + alpha * step, zeta, A, b, c);
      if (cand && Math.abs(cand.phi) < absPhi) {
        mu += alpha * step;
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!accepted) break;
  }

  const zFallback = fallbackPenalty(zeta, A, b, c, fallbackIters, tol);
  const qFallback = evaluateForm({ ...form, A }, zFallback);
  if (sense === "le" && qFallback <= tol) {
    return {
      z: zFallback,
      converged: false,
      iterations: maxNewtonIters,
      ...buildKktDiagnostics({
        zeta,
        z: zFallback,
        A,
        b,
        c,
        sense,
        tol,
      }),
    };
  }
  if (Math.abs(qFallback) <= (opts.tol ?? 1e-8) * 10) {
    return {
      z: zFallback,
      converged: false,
      iterations: maxNewtonIters,
      ...buildKktDiagnostics({
        zeta,
        z: zFallback,
        A,
        b,
        c,
        sense,
        tol,
      }),
    };
  }
  if (best) {
    return {
      z: best.z,
      converged: false,
      iterations: maxNewtonIters,
      ...buildKktDiagnostics({
        zeta,
        z: best.z,
        A,
        b,
        c,
        sense,
        tol,
      }),
    };
  }
  return {
    z: zeta.slice(),
    converged: false,
    iterations: maxNewtonIters,
    ...buildKktDiagnostics({
      zeta,
      z: zeta.slice(),
      A,
      b,
      c,
      sense,
      tol,
    }),
  };
}
