import type { Vec3 } from "../math/types";

export type PlaneEq = {
  n: Vec3;
  b: number;
};

export type PolyState = {
  vertices: Vec3[];
  faces: number[][];
  facePlanes: PlaneEq[];
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
