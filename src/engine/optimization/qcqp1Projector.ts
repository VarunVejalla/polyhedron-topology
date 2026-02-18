import type { ConstraintSense, DenseMatrix, IndexedQuadraticForm } from "./types";

type QCQP1ProjectOptions = {
  tol?: number;
  maxNewtonIters?: number;
  maxBacktrackSteps?: number;
  maxStepAbs?: number;
  fallbackIters?: number;
  cache?: QCQP1Cache;
  mu0?: number;
};

type QCQP1ProjectResult = {
  z: number[];
  converged: boolean;
  iterations: number;
  mu: number;
};

type QCQP1CacheKind = "generic" | "unit_sphere";

type Eigensystem = {
  Q: number[][];
  lambda: number[];
};

export type QCQP1Cache = {
  kind: QCQP1CacheKind;
  A: DenseMatrix;
  b: number[];
  c: number;
  eig?: Eigensystem;
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

function transposeMatVec(A: Readonly<DenseMatrix>, x: ReadonlyArray<number>): number[] {
  const out = new Array<number>(x.length).fill(0);
  for (let i = 0; i < A.length; i++) {
    const xi = x[i];
    const row = A[i];
    for (let j = 0; j < x.length; j++) out[j] += (row?.[j] ?? 0) * xi;
  }
  return out;
}

function evaluateForm(A: Readonly<DenseMatrix>, b: ReadonlyArray<number>, c: number, z: ReadonlyArray<number>): number {
  const Az = matVec(A, z);
  return 0.5 * dot(z, Az) + dot(b, z) + c;
}

function symmetrize(A: Readonly<DenseMatrix>): DenseMatrix {
  const n = A.length;
  const out = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) out[i][j] = 0.5 * ((A[i]?.[j] ?? 0) + (A[j]?.[i] ?? 0));
  }
  return out;
}

function jacobiEigenSym(A: Readonly<DenseMatrix>): Eigensystem {
  const n = A.length;
  const M: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => A[i]?.[j] ?? 0));
  const V: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  const maxSweeps = Math.max(20, 8 * n * n);
  const tol = 1e-12;

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let p = 0;
    let q = 1;
    let maxAbs = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const a = Math.abs(M[i][j]);
        if (a > maxAbs) {
          maxAbs = a;
          p = i;
          q = j;
        }
      }
    }
    if (maxAbs < tol) break;

    const app = M[p][p];
    const aqq = M[q][q];
    const apq = M[p][q];
    const theta = (aqq - app) / (2 * apq);
    const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    for (let k = 0; k < n; k++) {
      if (k === p || k === q) continue;
      const mkp = M[k][p];
      const mkq = M[k][q];
      M[k][p] = c * mkp - s * mkq;
      M[p][k] = M[k][p];
      M[k][q] = s * mkp + c * mkq;
      M[q][k] = M[k][q];
    }
    M[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    M[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    M[p][q] = 0;
    M[q][p] = 0;

    for (let k = 0; k < n; k++) {
      const vkp = V[k][p];
      const vkq = V[k][q];
      V[k][p] = c * vkp - s * vkq;
      V[k][q] = s * vkp + c * vkq;
    }
  }

  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => M[i][i] - M[j][j]);
  const lambda = order.map((idx) => M[idx][idx]);
  const Q = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let col = 0; col < n; col++) {
    const src = order[col];
    for (let row = 0; row < n; row++) Q[row][col] = V[row][src];
  }
  return { Q, lambda };
}

function isUnitSphereConstraint(A: Readonly<DenseMatrix>, b: ReadonlyArray<number>, c: number): boolean {
  const n = A.length;
  if (n === 0) return false;
  if (Math.abs(c + 1) > 1e-10) return false;
  for (let i = 0; i < n; i++) {
    if (Math.abs(b[i] ?? 0) > 1e-10) return false;
    for (let j = 0; j < n; j++) {
      const target = i === j ? 2 : 0;
      if (Math.abs((A[i]?.[j] ?? 0) - target) > 1e-10) return false;
    }
  }
  return true;
}

export function createQcqp1Cache(form: Readonly<IndexedQuadraticForm>): QCQP1Cache {
  const A = symmetrize(form.A);
  const b = form.b.slice();
  const c = form.c;
  if (isUnitSphereConstraint(A, b, c)) return { kind: "unit_sphere", A, b, c };
  return { kind: "generic", A, b, c, eig: jacobiEigenSym(A) };
}

function unitProjection(zeta: ReadonlyArray<number>, sense: ConstraintSense): number[] {
  const nrm = Math.sqrt(dot(zeta, zeta));
  if (sense === "le") {
    if (nrm <= 1) return zeta.slice();
    const inv = 1 / Math.max(1e-12, nrm);
    return zeta.map((v) => v * inv);
  }
  if (nrm <= 1e-12) {
    const out = new Array<number>(zeta.length).fill(0);
    out[0] = 1;
    return out;
  }
  const inv = 1 / nrm;
  return zeta.map((v) => v * inv);
}

function evalAtMuEig(
  mu: number,
  zeta: ReadonlyArray<number>,
  cache: Readonly<QCQP1Cache>
): { z: number[]; phi: number; phiPrime: number } | null {
  const eig = cache.eig;
  if (!eig) return null;
  const Qtzeta = transposeMatVec(eig.Q, zeta);
  const Qtb = transposeMatVec(eig.Q, cache.b);
  const n = Qtzeta.length;
  const ztilde = new Array<number>(n).fill(0);
  const vtilde = new Array<number>(n).fill(0);
  let phi = cache.c;
  let phiPrime = 0;
  for (let k = 0; k < n; k++) {
    const lam = eig.lambda[k];
    const denom = 1 + mu * lam;
    if (!Number.isFinite(denom) || Math.abs(denom) < 1e-12) return null;
    const zk = (Qtzeta[k] - mu * Qtb[k]) / denom;
    ztilde[k] = zk;
    phi += 0.5 * lam * zk * zk + Qtb[k] * zk;
    const vk = lam * zk + Qtb[k];
    vtilde[k] = vk;
    phiPrime -= (vk * vk) / denom;
  }
  if (!Number.isFinite(phi) || !Number.isFinite(phiPrime)) return null;
  const z = matVec(eig.Q, ztilde);
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
  const step = 0.08;
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
  const maxNewtonIters = opts.maxNewtonIters ?? 30;
  const maxBacktrackSteps = opts.maxBacktrackSteps ?? 12;
  const maxStepAbs = opts.maxStepAbs ?? 25;
  const fallbackIters = opts.fallbackIters ?? 40;
  const cache = opts.cache ?? createQcqp1Cache(form);

  if (cache.kind === "unit_sphere") {
    const z = unitProjection(zeta, sense);
    return { z, converged: true, iterations: 1, mu: 0 };
  }

  const q0 = evaluateForm(cache.A, cache.b, cache.c, zeta);
  if (sense === "le" && q0 <= tol) return { z: zeta.slice(), converged: true, iterations: 0, mu: 0 };

  let mu = opts.mu0 ?? 0;
  let best: { z: number[]; absPhi: number; mu: number } | null = null;

  for (let it = 0; it < maxNewtonIters; it++) {
    const cur = evalAtMuEig(mu, zeta, cache);
    if (!cur) break;
    const absPhi = Math.abs(cur.phi);
    if (!best || absPhi < best.absPhi) best = { z: cur.z, absPhi, mu };
    if (absPhi <= tol) return { z: cur.z, converged: true, iterations: it + 1, mu };

    const denom = Math.abs(cur.phiPrime) < 1e-14 ? (cur.phiPrime >= 0 ? 1e-14 : -1e-14) : cur.phiPrime;
    let step = -cur.phi / denom;
    if (step > maxStepAbs) step = maxStepAbs;
    if (step < -maxStepAbs) step = -maxStepAbs;
    if (!Number.isFinite(step)) break;

    let accepted = false;
    let alpha = 1;
    for (let bt = 0; bt < maxBacktrackSteps; bt++) {
      const muCand = mu + alpha * step;
      const cand = evalAtMuEig(muCand, zeta, cache);
      if (cand && Math.abs(cand.phi) < absPhi) {
        mu = muCand;
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!accepted) break;
  }

  const zFallback = fallbackPenalty(zeta, cache.A, cache.b, cache.c, fallbackIters, tol);
  const qFallback = evaluateForm(cache.A, cache.b, cache.c, zFallback);
  if (sense === "le" && qFallback <= tol) return { z: zFallback, converged: false, iterations: maxNewtonIters, mu };
  if (Math.abs(qFallback) <= tol * 10) return { z: zFallback, converged: false, iterations: maxNewtonIters, mu };
  if (best) return { z: best.z, converged: false, iterations: maxNewtonIters, mu: best.mu };
  return { z: zeta.slice(), converged: false, iterations: maxNewtonIters, mu };
}
