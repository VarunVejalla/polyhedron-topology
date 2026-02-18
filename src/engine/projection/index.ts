import type { Vec3 } from "../math/types";
import { PlanarProjector } from "./planarOptimizer";

export type ProjectionMethod = "planar" | "convex";

export type ProjectionFlavor = "planar" | "convex";

export const projectionMethods: { id: ProjectionMethod; label: string }[] = [
  { id: "planar", label: "Planar projection" },
  { id: "convex", label: "Planar + convex projection" },
];

export type HandleSet = {
  targets: Map<number, Vec3>;
};

export type ProjectorParams = {
  rho: number;
  wFree: number;
  wHandle: number;
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

export function createProjector(method: ProjectionMethod, faces: number[][], x0: Vec3[], params: ProjectorParams): IProjector {
  return new PlanarProjector(faces, x0, method, params);
}
