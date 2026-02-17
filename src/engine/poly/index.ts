export type {
  PlaneEq,
  PolyState,
  PolyRichState,
  PolyAuxState,
  PolyDerivedCache,
  PolyEdge,
  PolyTopologyData,
  FaceEdgeIncidence,
  VertexFaceIncidence,
  RollStep,
} from "./types";
export { buildPolyState, computeFacePlanes } from "./state";
export { buildPolyDerivedCache } from "./derive";
export { buildPolyTopology } from "./topology";
export { buildPolyAuxState, buildPolyRichState, computeVolumeAndCenterOfMass } from "./auxiliary";
export { computeAuxConstraintResiduals, type PolyConstraintResiduals, type ResidualBlock } from "./constraints";

