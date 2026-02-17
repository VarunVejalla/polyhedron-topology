export type {
  PlaneEq,
  PolyState,
  PolyRichState,
  PolyDerivedCache,
  PolyTopologyData,
  VertexFaceIncidence,
} from "./types";

export { buildPolyState } from "./state";
export {
  buildPolyLightModelFromState,
  lightVertexDim,
  lightFullDim,
  lightNBase,
  readLightVertex,
  packPolyLightState,
} from "./light";
export { buildPolyTopology } from "./topology";
export { buildPolyAuxState } from "./auxiliary";
export { buildPolyFullModel } from "./full";
export {
  incidenceConstraintValue,
  nonIncidenceConstraintValue,
} from "./lightConstraints";
