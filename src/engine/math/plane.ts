import type { Vec3 } from "./types";
import { v3 } from "./vec3";
import { smallestEigenvectorSym3 } from "./jacobi3";

export type Plane = {
  n: Vec3;
  b: number;
  c: Vec3;
  quality: number;
};

export function bestFitPlanePCA(pts: Vec3[], prevNormal?: Vec3): Plane {
  if (pts.length === 0) {
    const n: Vec3 = [0, 0, 1];
    return { n, b: 0, c: [0, 0, 0], quality: 0 };
  }

  let c: Vec3 = [0, 0, 0];
  for (let i = 0; i < pts.length; i++) c = v3.add(c, pts[i]);
  c = v3.mul(c, 1 / pts.length);

  let xx = 0;
  let xy = 0;
  let xz = 0;
  let yy = 0;
  let yz = 0;
  let zz = 0;
  for (let i = 0; i < pts.length; i++) {
    const q = v3.sub(pts[i], c);
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

  let n = v3.unit(smallestEigenvectorSym3(cov));
  if (prevNormal && v3.dot(n, prevNormal) < 0) n = v3.mul(n, -1);

  return {
    n,
    b: v3.dot(n, c),
    c,
    quality: xx + yy + zz,
  };
}

export function projectPointToPlane(p: Vec3, plane: Plane): Vec3 {
  return v3.sub(p, v3.mul(plane.n, v3.dot(plane.n, p) - plane.b));
}

export function planarityResiduals(pts: Vec3[], plane: Plane): { maxAbs: number; rms: number } {
  if (pts.length <= 3) return { maxAbs: 0, rms: 0 };
  let maxAbs = 0;
  let s2 = 0;
  for (let i = 0; i < pts.length; i++) {
    const d = v3.dot(plane.n, pts[i]) - plane.b;
    const ad = Math.abs(d);
    if (ad > maxAbs) maxAbs = ad;
    s2 += d * d;
  }
  return { maxAbs, rms: Math.sqrt(s2 / pts.length) };
}
