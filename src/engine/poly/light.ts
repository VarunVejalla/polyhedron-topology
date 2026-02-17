import type { Vec3 } from "../math/types";
import { buildPolyState } from "./state";
import { buildPolyTopology } from "./topology";
import type { PlaneEq, PolyState, PolyTopologyData } from "./types";

function dot3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export type PolyLightModel = {
  state: PolyState;
  topology: PolyTopologyData;
};

export type PolyLightConstraintMetrics = {
  planarityMetric: number;
  unitNormalityMetric: number;
  convexityViolation: number;
  isConvex: boolean;
};

export function buildPolyLightModel(
  vertices: ReadonlyArray<Vec3>,
  faces: ReadonlyArray<ReadonlyArray<number>>,
  prevPlanes?: ReadonlyArray<PlaneEq>
): PolyLightModel {
  const state = buildPolyState(vertices, faces, prevPlanes);
  const topology = buildPolyTopology(state.faces, state.vertices.length);
  return { state, topology };
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
