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
  lightBIndex,
  readLightVertex,
  packPolyLightState,
  unpackPolyLightState,
} from "./light";
export { buildPolyTopology } from "./topology";
export { buildPolyAuxState } from "./auxiliary";
export { buildPolyFullModel } from "./full";
export {
  incidenceConstraintValue,
  incidenceConstraintLinearization,
  nonIncidenceConstraintValue,
  squaredSlackNonIncidenceConstraintLinearization,
  unitNormalConstraintValue,
  unitNormalConstraintLinearization,
} from "./lightConstraints";
