import type { Vec3 } from "./math/types";
export type { Vec3 } from "./math/types";

type PrismState = {
  /** 6 vertices */
  vertices: Vec3[];
};

// Triangular prism topology (6 verts, 5 faces).
// Vertex ordering:
//   Top triangle:    0,1,2
//   Bottom triangle: 3,4,5 (corresponding to 0,1,2 respectively)
// Side faces are quads.
export const PRISM_FACES: number[][] = [
  [0, 1, 2],      // top
  [3, 5, 4],      // bottom (reversed for an outward-ish default)
  [0, 1, 4, 3],   // side
  [1, 2, 5, 4],   // side
  [2, 0, 3, 5],   // side
];

export function makeDefaultPrism(): PrismState {
  // A simple unit-ish prism.
  const h = 1.2;
  const r = 0.85;
  const a0 = 0;
  const a1 = (2 * Math.PI) / 3;
  const a2 = (4 * Math.PI) / 3;
  const top: Vec3[] = [
    [r * Math.cos(a0), r * Math.sin(a0), h / 2],
    [r * Math.cos(a1), r * Math.sin(a1), h / 2],
    [r * Math.cos(a2), r * Math.sin(a2), h / 2],
  ];
  const bot: Vec3[] = top.map(([x, y]) => [x, y, -h / 2]);
  return { vertices: [...top, ...bot] as Vec3[] };
}
