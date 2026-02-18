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
    return { z: zeta.slice(), converged: true, iterations: 0 };
  }

  let mu = 0;
  let best: { z: number[]; absPhi: number } | null = null;

  for (let it = 0; it < maxNewtonIters; it++) {
    const cur = evalAtMu(mu, zeta, A, b, c);
    if (!cur) break;
    const absPhi = Math.abs(cur.phi);
    if (!best || absPhi < best.absPhi) best = { z: cur.z, absPhi };
    if (absPhi <= tol) return { z: cur.z, converged: true, iterations: it + 1 };

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
  if (sense === "le" && qFallback <= tol) return { z: zFallback, converged: false, iterations: maxNewtonIters };
  if (Math.abs(qFallback) <= (opts.tol ?? 1e-8) * 10) return { z: zFallback, converged: false, iterations: maxNewtonIters };
  if (best) return { z: best.z, converged: false, iterations: maxNewtonIters };
  return { z: zeta.slice(), converged: false, iterations: maxNewtonIters };
}
