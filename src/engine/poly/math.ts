import type { Vec3 } from "../math/types";

export function dot3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function sub3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function cross3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function averageVertices(vertices: ReadonlyArray<Vec3>, ids?: ReadonlyArray<number>): Vec3 {
  const c: Vec3 = [0, 0, 0];
  const n = ids ? ids.length : vertices.length;
  if (n === 0) return c;
  if (ids) {
    for (let i = 0; i < ids.length; i++) {
      const p = vertices[ids[i]];
      c[0] += p[0];
      c[1] += p[1];
      c[2] += p[2];
    }
  } else {
    for (let i = 0; i < vertices.length; i++) {
      c[0] += vertices[i][0];
      c[1] += vertices[i][1];
      c[2] += vertices[i][2];
    }
  }
  const inv = 1 / n;
  return [c[0] * inv, c[1] * inv, c[2] * inv];
}
