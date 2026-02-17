import type { Vec3 } from "../math/types";
import { buildPolyTopology } from "./topology";
import type { PlaneEq, PolyState, PolyTopologyData } from "./types";

function dot3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

type PolyLightModel = {
  state: PolyState;
  topology: PolyTopologyData;
};

type PolyLightConstraintMetrics = {
  planarityMetric: number;
  unitNormalityMetric: number;
  convexityViolation: number;
  isConvex: boolean;
};

export function lightVertexDim(vertexCount: number): number {
  return 3 * vertexCount;
}

export function lightFullDim(vertexCount: number, faceCount: number): number {
  return lightVertexDim(vertexCount) + 4 * faceCount;
}

export function lightNBase(vertexCount: number, fi: number): number {
  return lightVertexDim(vertexCount) + 4 * fi;
}

export function lightBIndex(vertexCount: number, fi: number): number {
  return lightNBase(vertexCount, fi) + 3;
}

export function readLightVertex(y: ReadonlyArray<number>, vi: number): Vec3 {
  const b = 3 * vi;
  return [y[b], y[b + 1], y[b + 2]];
}

export function readLightNormal(y: ReadonlyArray<number>, vertexCount: number, fi: number): Vec3 {
  const nb = lightNBase(vertexCount, fi);
  return [y[nb], y[nb + 1], y[nb + 2]];
}

export function readLightOffset(y: ReadonlyArray<number>, vertexCount: number, fi: number): number {
  return y[lightBIndex(vertexCount, fi)];
}

export function packPolyLightState(state: PolyState): number[] {
  const out = new Array<number>(lightFullDim(state.vertices.length, state.faces.length));
  for (let i = 0; i < state.vertices.length; i++) {
    const b = 3 * i;
    out[b] = state.vertices[i][0];
    out[b + 1] = state.vertices[i][1];
    out[b + 2] = state.vertices[i][2];
  }
  for (let fi = 0; fi < state.faces.length; fi++) {
    const nb = lightNBase(state.vertices.length, fi);
    const pl = state.facePlanes[fi];
    out[nb] = pl.n[0];
    out[nb + 1] = pl.n[1];
    out[nb + 2] = pl.n[2];
    out[nb + 3] = pl.b;
  }
  return out;
}

export function unpackPolyLightState(
  y: ReadonlyArray<number>,
  faces: ReadonlyArray<ReadonlyArray<number>>,
  vertexCount: number
): PolyState {
  const vertices: Vec3[] = new Array(vertexCount);
  const facePlanes: PlaneEq[] = new Array(faces.length);
  for (let i = 0; i < vertexCount; i++) {
    const b = 3 * i;
    vertices[i] = [y[b], y[b + 1], y[b + 2]];
  }
  for (let fi = 0; fi < faces.length; fi++) {
    const nb = lightNBase(vertexCount, fi);
    facePlanes[fi] = {
      n: [y[nb], y[nb + 1], y[nb + 2]],
      b: y[nb + 3],
    };
  }
  return {
    vertices,
    faces: faces.map((f) => [...f]),
    facePlanes,
  };
}

export function buildPolyLightModelFromState(state: PolyState): PolyLightModel {
  return {
    state: {
      vertices: state.vertices.map((p) => [p[0], p[1], p[2]]),
      faces: state.faces.map((f) => [...f]),
      facePlanes: state.facePlanes.map((pl) => ({ n: [pl.n[0], pl.n[1], pl.n[2]], b: pl.b })),
    },
    topology: buildPolyTopology(state.faces, state.vertices.length),
  };
}

export function computePolyLightConstraintMetrics(state: PolyState): PolyLightConstraintMetrics {
  let planarityMetric = 0;
  for (let fi = 0; fi < state.faces.length; fi++) {
    const pl = state.facePlanes[fi];
    for (let li = 0; li < state.faces[fi].length; li++) {
      const vi = state.faces[fi][li];
      const p = state.vertices[vi];
      const d = dot3(pl.n, p) - pl.b;
      planarityMetric += d * d;
    }
  }

  let unitNormalityMetric = 0;
  for (let fi = 0; fi < state.facePlanes.length; fi++) {
    const n = state.facePlanes[fi].n;
    const d = dot3(n, n) - 1;
    unitNormalityMetric += d * d;
  }

  let convexityViolation = 0;
  let isConvex = true;
  const incidence = state.faces.map((f) => new Set<number>(f));
  for (let fi = 0; fi < state.faces.length; fi++) {
    const pl = state.facePlanes[fi];
    for (let vi = 0; vi < state.vertices.length; vi++) {
      if (incidence[fi].has(vi)) continue;
      const v = dot3(pl.n, state.vertices[vi]) - pl.b;
      if (v > 0) {
        isConvex = false;
        convexityViolation += v * v;
      }
    }
  }

  return {
    planarityMetric,
    unitNormalityMetric,
    convexityViolation,
    isConvex,
  };
}
