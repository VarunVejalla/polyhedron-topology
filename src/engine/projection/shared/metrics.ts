import { bestFitPlanePCA } from "../../geom/plane";
import type { Vec3 } from "../../math/types";

export function sumSquaredPlanarityResidual(
  faces: ReadonlyArray<ReadonlyArray<number>>,
  positions: ReadonlyArray<Vec3>
): number {
  let total = 0;
  for (let fi = 0; fi < faces.length; fi++) {
    const face = faces[fi];
    if (face.length < 3) continue;
    const pts = face.map((vi) => positions[vi]);
    const plane = bestFitPlanePCA(pts);
    const n = plane.n;
    const b = plane.b;
    for (let k = 0; k < face.length; k++) {
      const p = positions[face[k]];
      const d = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - b;
      total += d * d;
    }
  }
  return total;
}
