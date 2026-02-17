import type { Vec3 } from "../math/types";
import { PlanarProjector } from "./planarOptimizer";

export type ProjectionMethod =
  | "admm"
  | "admm_convex"
  | "admm_regular"
  | "guided_alm"
  | "guided_alm_squared_slack"
  | "guided_alm_modular";

export type ProjectionFlavor = "planar" | "convex" | "regular";

export const projectionMethods: { id: ProjectionMethod; label: string }[] = [
  { id: "admm", label: "Planar projection" },
  { id: "admm_convex", label: "Planar + convex projection" },
  { id: "admm_regular", label: "Planar + regularized projection" },
  { id: "guided_alm", label: "Planar + regularized projection (alias)" },
  { id: "guided_alm_squared_slack", label: "Planar + convex projection (alias)" },
  { id: "guided_alm_modular", label: "Planar + regularized projection (alias)" },
];

export type HandleSet = {
  targets: Map<number, Vec3>;
};

export type ProjectorParams = {
  rho: number;
  wFree: number;
  wHandle: number;
  lambdaReg: number;
  itersPerFrame: number;
  itersOnRelease: number;
};

export interface IProjector {
  // Topology is immutable after creation; changing it requires creating a new projector.
  reset(x0: Vec3[]): void;
  setBaseline(x0: Vec3[]): void;
  setHandles(handles: HandleSet): void;
  step(iterations: number): void;
  getPositionsRef(): ReadonlyArray<Vec3>;
  snapshotPositions(): Vec3[];
  diagnostics(): { totalPlanarityViolation: number };
  setParams?(next: Partial<ProjectorParams>): void;
}

function methodFlavor(method: ProjectionMethod): ProjectionFlavor {
  if (method === "admm_convex" || method === "guided_alm_squared_slack") return "convex";
  if (method === "admm_regular" || method === "guided_alm" || method === "guided_alm_modular") return "regular";
  return "planar";
}

export function createProjector(method: ProjectionMethod, faces: number[][], x0: Vec3[], params: ProjectorParams): IProjector {
  return new PlanarProjector(faces, x0, methodFlavor(method), params);
}
