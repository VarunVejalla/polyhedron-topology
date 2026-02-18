import { bestFitPlanePCA, type Plane } from "../math/plane";
import type { Vec3 } from "../math/types";
import { v3 } from "../math/vec3";

type VertexIncidence = Array<Array<{ fi: number; li: number }>>;

export function buildVertexIncidence(
  faces: ReadonlyArray<ReadonlyArray<number>>,
  vertexCount: number
): VertexIncidence {
  const inc: VertexIncidence = Array.from({ length: vertexCount }, () => []);
  for (let fi = 0; fi < faces.length; fi++) {
    const f = faces[fi];
    for (let li = 0; li < f.length; li++) inc[f[li]].push({ fi, li });
  }
  return inc;
}

export function createPlanarFaceBuffers(
  faces: ReadonlyArray<ReadonlyArray<number>>,
  x: ReadonlyArray<Vec3>
): { y: Vec3[][]; u: Vec3[][]; vbuf: Vec3[][] } {
  return {
    y: faces.map((f) => f.map((vi) => [...x[vi]] as Vec3)),
    u: faces.map((f) => f.map(() => [0, 0, 0] as Vec3)),
    vbuf: faces.map((f) => f.map(() => [0, 0, 0] as Vec3)),
  };
}

export function fitFacePlanesFromPositions(
  faces: ReadonlyArray<ReadonlyArray<number>>,
  x: ReadonlyArray<Vec3>,
  prevFaceNormals: Array<Vec3 | undefined>
): Plane[] {
  const planes = new Array<Plane>(faces.length);
  for (let fi = 0; fi < faces.length; fi++) {
    const pts = faces[fi].map((vi) => x[vi]);
    const plane = bestFitPlanePCA(pts, prevFaceNormals[fi]);
    prevFaceNormals[fi] = plane.n;
    planes[fi] = plane;
  }
  return planes;
}

export function updatePlanarYBlock(
  faces: ReadonlyArray<ReadonlyArray<number>>,
  x: ReadonlyArray<Vec3>,
  u: Vec3[][],
  vbuf: Vec3[][],
  y: Vec3[][],
  prevFaceNormals: Array<Vec3 | undefined>,
  facePlanes: Plane[]
) {
  for (let fi = 0; fi < faces.length; fi++) {
    const f = faces[fi];
    const v = vbuf[fi];
    for (let li = 0; li < f.length; li++) {
      const vi = f[li];
      const px = x[vi];
      const uu = u[fi][li];
      v[li] = v3.add(px, uu);
    }

    const plane = bestFitPlanePCA(v, prevFaceNormals[fi]);
    prevFaceNormals[fi] = plane.n;
    facePlanes[fi] = plane;

    const yfi = y[fi];
    const n = plane.n;
    const b = plane.b;
    for (let li = 0; li < f.length; li++) {
      const p = v[li];
      const t = v3.dot(n, p) - b;
      yfi[li] = v3.sub(p, v3.mul(n, t));
    }
  }
}

export function updatePlanarDualBlock(
  faces: ReadonlyArray<ReadonlyArray<number>>,
  x: ReadonlyArray<Vec3>,
  y: Vec3[][],
  u: Vec3[][]
) {
  for (let fi = 0; fi < faces.length; fi++) {
    const f = faces[fi];
    for (let li = 0; li < f.length; li++) {
      const vi = f[li];
      const uu = u[fi][li];
      const px = x[vi];
      const py = y[fi][li];
      u[fi][li] = v3.add(uu, v3.sub(px, py));
    }
  }
}

export function computePlanarityViolationFromPlanes(
  faces: ReadonlyArray<ReadonlyArray<number>>,
  positions: ReadonlyArray<Vec3>,
  facePlanes: ReadonlyArray<Plane>
): number {
  let total = 0;
  for (let fi = 0; fi < faces.length; fi++) {
    const face = faces[fi];
    const plane = facePlanes[fi];
    const n = plane.n;
    const b = plane.b;
    for (let k = 0; k < face.length; k++) {
      const p = positions[face[k]];
      const d = v3.dot(n, p) - b;
      total += Math.abs(d);
    }
  }
  return total;
}
