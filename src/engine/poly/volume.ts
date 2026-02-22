import type { Vec3 } from "../math/types";
import { v3 } from "../math/vec3";
import type { PlaneEq } from "./types";
import { buildPolyState } from "./state";

function computeSignedVolumeFromVerticesAndPlanes(
  vertices: ReadonlyArray<Vec3>,
  faces: ReadonlyArray<ReadonlyArray<number>>,
  planes: ReadonlyArray<PlaneEq>
): number {
  let volume = 0;
  for (let fi = 0; fi < faces.length; fi++) {
    const face = faces[fi];
    const plane = planes[fi];
    let vectorArea: Vec3 = [0, 0, 0];
    for (let i = 0; i < face.length; i++) {
      const a = vertices[face[i]];
      const b = vertices[face[(i + 1) % face.length]];
      vectorArea = v3.add(vectorArea, v3.mul(v3.cross(a, b), 0.5));
    }
    volume += (plane.b * v3.dot(plane.n, vectorArea)) / 3;
  }
  return volume;
}

export function computeSignedVolumeFromVerticesAndFaces(
  vertices: ReadonlyArray<Vec3>,
  faces: ReadonlyArray<ReadonlyArray<number>>
): number {
  const state = buildPolyState(vertices, faces);
  return computeSignedVolumeFromVerticesAndPlanes(state.vertices, state.faces, state.facePlanes);
}
