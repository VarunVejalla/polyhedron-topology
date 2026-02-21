import type { ConstraintSense, DenseMatrix } from "./types";
import {
  QCQP_BISECTION_ITERS,
  QCQP_BRACKET_EXPANSIONS,
  QCQP_DENOM_EPS,
  QCQP_EIG_TOL,
  QCQP_ROOT_EPS,
  QCQP_ROOT_WIDTH_EPS,
  SOLVER_PIVOT_EPS,
} from "../math/constants";

export type Qcqp1Constraint = {
  sense: ConstraintSense;
  A: DenseMatrix;
  b: number[];
  c: number;
};

type Qcqp1Result = {
  z: number[];
  converged: boolean;
  mu?: number;
};

function dot(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  let out = 0;
  for (let i = 0; i < a.length; i++) out += a[i] * b[i];
  return out;
}

function mulMatVec(A: ReadonlyArray<ReadonlyArray<number>>, x: ReadonlyArray<number>): number[] {
  const out = new Array<number>(x.length).fill(0);
  for (let i = 0; i < A.length; i++) {
    let s = 0;
    const row = A[i] ?? [];
    for (let j = 0; j < x.length; j++) s += (row[j] ?? 0) * x[j];
    out[i] = s;
  }
  return out;
}

function mulMatTVec(A: ReadonlyArray<ReadonlyArray<number>>, x: ReadonlyArray<number>): number[] {
  const n = x.length;
  const out = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += (A[j]?.[i] ?? 0) * x[j];
    out[i] = s;
  }
  return out;
}

function symmetrize(A: ReadonlyArray<ReadonlyArray<number>>): number[][] {
  const n = A.length;
  const out = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const v = 0.5 * ((A[i]?.[j] ?? 0) + (A[j]?.[i] ?? 0));
      out[i][j] = v;
      out[j][i] = v;
    }
  }
  return out;
}

function identity(n: number): number[][] {
  const out = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) out[i][i] = 1;
  return out;
}

function jacobiEigenSymmetric(Ain: ReadonlyArray<ReadonlyArray<number>>): { values: number[]; Q: number[][] } {
  const n = Ain.length;
  const A = symmetrize(Ain);
  const Q = identity(n);
  const maxSweeps = Math.max(20, 8 * n * n);
  const tol = QCQP_EIG_TOL;

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let p = 0;
    let q = 1;
    let maxOff = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const v = Math.abs(A[i][j]);
        if (v > maxOff) {
          maxOff = v;
          p = i;
          q = j;
        }
      }
    }
    if (maxOff < tol) break;

    const app = A[p][p];
    const aqq = A[q][q];
    const apq = A[p][q];
    if (Math.abs(apq) < tol) continue;

    const tau = (aqq - app) / (2 * apq);
    const t = tau >= 0 ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    for (let k = 0; k < n; k++) {
      if (k === p || k === q) continue;
      const aik = A[k][p];
      const akq = A[k][q];
      A[k][p] = c * aik - s * akq;
      A[p][k] = A[k][p];
      A[k][q] = s * aik + c * akq;
      A[q][k] = A[k][q];
    }
    const appNew = c * c * app - 2 * s * c * apq + s * s * aqq;
    const aqqNew = s * s * app + 2 * s * c * apq + c * c * aqq;
    A[p][p] = appNew;
    A[q][q] = aqqNew;
    A[p][q] = 0;
    A[q][p] = 0;

    for (let k = 0; k < n; k++) {
      const qkp = Q[k][p];
      const qkq = Q[k][q];
      Q[k][p] = c * qkp - s * qkq;
      Q[k][q] = s * qkp + c * qkq;
    }
  }

  const values = new Array<number>(n);
  for (let i = 0; i < n; i++) values[i] = A[i][i];
  return { values, Q };
}

function evalConstraint(constraint: Qcqp1Constraint, z: ReadonlyArray<number>): number {
  const Az = mulMatVec(constraint.A, z);
  return dot(z, Az) - 2 * dot(constraint.b, z) - constraint.c;
}

function solveEquality(zeta: ReadonlyArray<number>, c: Qcqp1Constraint): Qcqp1Result {
  const eig = jacobiEigenSymmetric(c.A);
  const lam = eig.values;
  const zetaP = mulMatTVec(eig.Q, zeta);
  const bP = mulMatTVec(eig.Q, c.b);

  const phi = (mu: number): number => {
    let out = -c.c;
    for (let k = 0; k < lam.length; k++) {
      const denom = 1 + mu * lam[k];
      if (Math.abs(denom) < QCQP_DENOM_EPS) return Number.NaN;
      const zk = (zetaP[k] + mu * bP[k]) / denom;
      out += lam[k] * zk * zk - 2 * bP[k] * zk;
    }
    return out;
  };

  let lower = -Infinity;
  let upper = Infinity;
  for (let k = 0; k < lam.length; k++) {
    const lk = lam[k];
    if (lk > SOLVER_PIVOT_EPS) lower = Math.max(lower, -1 / lk);
    if (lk < -SOLVER_PIVOT_EPS) upper = Math.min(upper, -1 / lk);
  }
  const eps = QCQP_ROOT_EPS;
  let lo = Number.isFinite(lower) ? lower + eps : -1;
  let hi = Number.isFinite(upper) ? upper - eps : 1;

  let flo = phi(lo);
  let fhi = phi(hi);
  for (let expand = 0; expand < QCQP_BRACKET_EXPANSIONS && (!Number.isFinite(flo) || !Number.isFinite(fhi) || flo * fhi > 0); expand++) {
    if (!Number.isFinite(lower)) lo *= 2;
    if (!Number.isFinite(upper)) hi *= 2;
    flo = phi(lo);
    fhi = phi(hi);
    if (Number.isFinite(lower) && Number.isFinite(upper)) break;
  }

  if (!Number.isFinite(flo) || !Number.isFinite(fhi) || flo * fhi > 0) return { z: [...zeta], converged: false };

  let a = lo;
  let b = hi;
  let fa = flo;
  let fb = fhi;
  let mu = 0.5 * (a + b);
  let converged = false;
  for (let it = 0; it < QCQP_BISECTION_ITERS; it++) {
    mu = 0.5 * (a + b);
    const fm = phi(mu);
    if (!Number.isFinite(fm)) {
      converged = false;
      break;
    }
    if (Math.abs(fm) < QCQP_ROOT_EPS || Math.abs(b - a) < QCQP_ROOT_WIDTH_EPS) {
      converged = true;
      break;
    }
    if (fa * fm <= 0) {
      b = mu;
      fb = fm;
    } else {
      a = mu;
      fa = fm;
    }
    if (!Number.isFinite(fa) || !Number.isFinite(fb)) break;
  }

  const zP = new Array<number>(lam.length);
  for (let k = 0; k < lam.length; k++) {
    const denom = 1 + mu * lam[k];
    if (Math.abs(denom) < QCQP_DENOM_EPS) return { z: [...zeta], converged: false };
    zP[k] = (zetaP[k] + mu * bP[k]) / denom;
  }
  const z = mulMatVec(eig.Q, zP);
  return { z, converged, mu };
}

export function projectQcqp1Paper(zeta: ReadonlyArray<number>, constraint: Qcqp1Constraint): Qcqp1Result {
  if (constraint.sense === "le" && evalConstraint(constraint, zeta) <= QCQP_ROOT_EPS) return { z: [...zeta], converged: true };
  return solveEquality(zeta, constraint);
}
