import { EPS } from "./constants";
import type { Vec3 } from "./types";

export const v3 = {
  add: (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  mul: (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a: Readonly<Vec3>, b: Readonly<Vec3>): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ],
  norm2: (a: Readonly<Vec3>): number => a[0] * a[0] + a[1] * a[1] + a[2] * a[2],
  norm: (a: Readonly<Vec3>): number => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]),
  unit: (a: Readonly<Vec3>): Vec3 => {
    const n = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    if (n < EPS) return [0, 0, 1];
    return [a[0] / n, a[1] / n, a[2] / n];
  },
  lerp: (a: Readonly<Vec3>, b: Readonly<Vec3>, t: number): Vec3 => [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ],
  clamp01: (t: number): number => Math.max(0, Math.min(1, t)),
};
