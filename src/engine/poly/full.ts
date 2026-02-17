import { buildPolyAuxState } from "./auxiliary";
import { computeAuxConstraintResiduals, type PolyConstraintResiduals } from "./constraints";
import { buildPolyDerivedCache } from "./derive";
import { buildPolyTopology } from "./topology";
import type { PolyAuxState, PolyDerivedCache, PolyRichState, PolyState, PolyTopologyData } from "./types";

type PolyFullModel = {
  rich: PolyRichState;
  derived: PolyDerivedCache;
  constraintResiduals?: PolyConstraintResiduals;
};

function asRichState(
  state: PolyState | PolyRichState,
  topologyArg?: PolyTopologyData,
  auxArg?: PolyAuxState
): PolyRichState {
  if ("topology" in state && "aux" in state) return state;
  const topology = topologyArg ?? buildPolyTopology(state.faces, state.vertices.length);
  const aux = auxArg ?? buildPolyAuxState(state, topology);
  return {
    vertices: state.vertices.map((p) => [p[0], p[1], p[2]]),
    faces: state.faces.map((f) => [...f]),
    facePlanes: state.facePlanes.map((pl) => ({ n: [pl.n[0], pl.n[1], pl.n[2]], b: pl.b })),
    topology,
    aux,
  };
}

export function buildPolyFullModel(
  state: PolyState | PolyRichState,
  topologyArg?: PolyTopologyData,
  auxArg?: PolyAuxState,
  options?: { includeConstraintResiduals?: boolean }
): PolyFullModel {
  const rich = asRichState(state, topologyArg, auxArg);
  const derived = buildPolyDerivedCache(rich);
  const constraintResiduals = options?.includeConstraintResiduals
    ? computeAuxConstraintResiduals(rich, rich.aux, rich.topology)
    : undefined;
  return { rich, derived, constraintResiduals };
}
