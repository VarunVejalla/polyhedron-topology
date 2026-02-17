import { bestFitPlanePCA } from "../math/plane";
import type { Vec3 } from "../math/types";
import { v3 } from "../math/vec3";
import type { PlaneEq, PolyState } from "./types";

function averageVertices(vertices: ReadonlyArray<Vec3>): Vec3 {
  const c: Vec3 = [0, 0, 0];
  if (vertices.length === 0) return c;
  for (let i = 0; i < vertices.length; i++) {
    c[0] += vertices[i][0];
    c[1] += vertices[i][1];
    c[2] += vertices[i][2];
  }
  return v3.mul(c, 1 / vertices.length);
}

function computeFacePlanes(
  vertices: ReadonlyArray<Vec3>,
  faces: ReadonlyArray<ReadonlyArray<number>>,
  prevPlanes?: ReadonlyArray<PlaneEq>
): PlaneEq[] {
  const interior = averageVertices(vertices);
  const out: PlaneEq[] = new Array(faces.length);

  for (let fi = 0; fi < faces.length; fi++) {
    const pts = faces[fi].map((vi) => vertices[vi]);
    const plane = bestFitPlanePCA(pts, prevPlanes?.[fi]?.n);
    let n: Vec3 = [plane.n[0], plane.n[1], plane.n[2]];
    let b = plane.b;
    if (v3.dot(n, interior) - b > 0) {
      n = [-n[0], -n[1], -n[2]];
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
  const f = faces.map((face) => [...face]);
  return { vertices: v, faces: f, facePlanes: computeFacePlanes(v, f, prevPlanes) };
}
