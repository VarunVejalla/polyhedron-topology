import type { Vec3 } from "../math/types";
import { v3 } from "../math/vec3";
import { smallestEigenvectorSym3 } from "../math/jacobi3";

export type Plane = {
  /** Unit normal (deterministic up to sign; use orientPlaneNormal for continuity). */
  n: Vec3;
  /** Offset such that n·x = b for any x on the plane. */
  b: number;
  /** Centroid of the input points (useful for diagnostics). */
  c: Vec3;
  /** A small degeneracy score: smallest eigenvalue scale proxy (0 ~= degenerate). */
  quality: number;
};

/**
 * Best-fit plane via PCA: normal is eigenvector of smallest eigenvalue of covariance.
 * Returns plane in the form n·x = b with n unit.
 */
export function bestFitPlanePCA(pts: Vec3[], prevNormal?: Vec3): Plane {
  if (pts.length === 0) {
    const n: Vec3 = [0, 0, 1];
    return { n, b: 0, c: [0, 0, 0], quality: 0 };
  }

  // centroid
  let c: Vec3 = [0, 0, 0];
  for (const p of pts) c = v3.add(c, p);
  c = v3.mul(c, 1 / pts.length);

  // covariance of centered points
  let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
  for (const p of pts) {
    const q = v3.sub(p, c);
    xx += q[0] * q[0];
    xy += q[0] * q[1];
    xz += q[0] * q[2];
    yy += q[1] * q[1];
    yz += q[1] * q[2];
    zz += q[2] * q[2];
  }
  const cov = [
    [xx, xy, xz],
    [xy, yy, yz],
    [xz, yz, zz],
  ];

  let n = smallestEigenvectorSym3(cov);
  n = v3.unit(n);

  // Stabilize sign if we have a previous normal.
  if (prevNormal && v3.dot(n, prevNormal) < 0) n = v3.mul(n, -1);

  const b = v3.dot(n, c);

  // quality heuristic: trace magnitude (0 means all points equal / nearly so)
  const trace = xx + yy + zz;
  const quality = trace;

  return { n, b, c, quality };
}

export function projectPointToPlane(p: Vec3, plane: Plane): Vec3 {
  const t = v3.dot(plane.n, p) - plane.b;
  return v3.sub(p, v3.mul(plane.n, t));
}

type PlanarityResiduals = {
  maxAbs: number;
  rms: number;
};

/** Planarity residuals relative to a plane n·x=b. */
export function planarityResiduals(pts: Vec3[], plane: Plane): PlanarityResiduals {
  if (pts.length <= 3) return { maxAbs: 0, rms: 0 };
  let maxAbs = 0;
  let s2 = 0;
  for (const p of pts) {
    const d = v3.dot(plane.n, p) - plane.b;
    const ad = Math.abs(d);
    if (ad > maxAbs) maxAbs = ad;
    s2 += d * d;
  }
  return { maxAbs, rms: Math.sqrt(s2 / pts.length) };
}
