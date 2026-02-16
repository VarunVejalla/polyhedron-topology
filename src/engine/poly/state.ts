import { bestFitPlanePCA } from "../geom/plane";
import type { Vec3 } from "../math/types";
import type { PlaneEq, PolyState } from "./types";

function dot3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function negate3(v: ReadonlyArray<number>): Vec3 {
  return [-v[0], -v[1], -v[2]];
}

export function computeFacePlanes(
  vertices: ReadonlyArray<Vec3>,
  faces: ReadonlyArray<ReadonlyArray<number>>,
  prevPlanes?: ReadonlyArray<PlaneEq>
): PlaneEq[] {
  const interior: Vec3 = [0, 0, 0];
  for (let i = 0; i < vertices.length; i++) {
    interior[0] += vertices[i][0];
    interior[1] += vertices[i][1];
    interior[2] += vertices[i][2];
  }
  if (vertices.length > 0) {
    const inv = 1 / vertices.length;
    interior[0] *= inv;
    interior[1] *= inv;
    interior[2] *= inv;
  }

  const out: PlaneEq[] = new Array(faces.length);
  for (let fi = 0; fi < faces.length; fi++) {
    const pts = faces[fi].map((vi) => vertices[vi]);
    const prevN = prevPlanes?.[fi]?.n;
    const plane = bestFitPlanePCA(pts, prevN);
    let n: Vec3 = [plane.n[0], plane.n[1], plane.n[2]];
    let b = plane.b;
    // Outward orientation: interior point must satisfy n·x <= b.
    const side = dot3(n, interior) - b;
    if (side > 0) {
      n = negate3(n);
      b = -b;
    }
    out[fi] = { n, b };
  }
  return out;
}

export function buildPolyState(
  vertices: ReadonlyArray<Vec3>,
  faces: ReadonlyArray<ReadonlyArray<number>>,
  prevPlanes?: ReadonlyArray<PlaneEq>
): PolyState {
  const v = vertices.map((p) => [p[0], p[1], p[2]] as Vec3);
  const f = faces.map((cy) => [...cy]);
  const facePlanes = computeFacePlanes(v, f, prevPlanes);
  return { vertices: v, faces: f, facePlanes };
}

