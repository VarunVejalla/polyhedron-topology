import type { Vec3 } from "../math/types";

export type PlaneEq = {
  n: Vec3;
  b: number;
};

export type PolyEdge = {
  edgeIndex: number;
  a: number;
  b: number;
};

export type FaceEdgeIncidence = {
  fi: number;
  localEdge: number;
  from: number;
  to: number;
  edgeIndex: number;
  edgeA: number;
  edgeB: number;
  sign: -1 | 1;
};

export type VertexFaceIncidence = {
  fi: number;
  vi: number;
};

export type PolyTopologyData = {
  vertexCount: number;
  faces: number[][];
  edges: PolyEdge[];
  edgeIncidencesByFace: FaceEdgeIncidence[][];
  edgeIncidencesFlat: FaceEdgeIncidence[];
  incidencePairs: VertexFaceIncidence[];
  nonIncidencePairs: VertexFaceIncidence[];
};

export type PolyState = {
  vertices: Vec3[];
  faces: number[][];
  facePlanes: PlaneEq[];
};

export type PolyAuxState = {
  // T_e for each undirected edge (edge canonical orientation).
  edgeCross: Vec3[];
  // a_k, A_k, R_k, V
  faceVectorArea: Vec3[];
  faceScalarArea: number[];
  facePyramidVolume: number[];
  volume: number;
  // c_k, C, P_k, d_k
  faceCentroid: Vec3[];
  centerOfMass: Vec3;
  projectedComByFace: Vec3[];
  faceComDistance: number[];
  // A_{f,i,j} aligned with local face edges.
  signedIncidenceArea: number[][];
};

export type PolyRichState = PolyState & {
  topology: PolyTopologyData;
  aux: PolyAuxState;
};

export type RollStep = {
  nextFace: number | null;
  edge: [number, number] | null;
};

export type PolyDerivedCache = {
  planarityMetric: number;
  unitNormalityMetric: number;
  convexityViolation: number;
  isConvex: boolean;
  centerOfMass: Vec3;
  volume: number;
  projectedComByFace: Vec3[];
  stableFace: boolean[];
  signedMarginByFace: number[];
  nextRollByFace: RollStep[];
  settleFaceByFace: Array<number | null>;
  settlePathByFace: number[][];
  defaultColorByFace: number[];
  basinIdByFace: number[];
  basinColorByFace: number[];
  basinFaces: Map<number, number[]>;
};
