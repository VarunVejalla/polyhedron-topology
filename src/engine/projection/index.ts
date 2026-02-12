import type { Vec3 } from "../math/types";
import { ADMMPlanarProjector, ADMMParams } from "./planarAdmm";
import { ADMMConvexPlanarProjector, ADMMConvexParams } from "./planarAdmmConvex";
import { AlternatingPlanarProjector, AlternatingParams } from "./planarAlternating";

export type ProjectionMethod = "admm" | "admm_convex" | "alternating";

export const projectionMethods: { id: ProjectionMethod; label: string }[] = [
  { id: "admm", label: "ADMM / prox (planar faces)" },
  { id: "admm_convex", label: "ADMM / prox (planar + convex)" },
  { id: "alternating", label: "Alternating LS (planes then vertices)" },
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
  setHandles(handles: HandleSet): void;
  step(iterations: number): void;
  getPositionsRef(): ReadonlyArray<Vec3>;
  snapshotPositions(): Vec3[];
  diagnostics(): { totalPlanarityViolation: number };
  // optional runtime param updates
  setParams?(next: ADMMParams | ADMMConvexParams | AlternatingParams): void;
}

export function createProjector(method: ProjectionMethod, faces: number[][], x0: Vec3[], params: ProjectorParams): IProjector {
  if (method === "alternating") {
    const p: AlternatingParams = { wFree: params.wFree, wHandle: params.wHandle };
    return new AlternatingPlanarProjector(faces, x0, p);
  }
  if (method === "admm_convex") {
    const p: ADMMConvexParams = { rho: params.rho, wFree: params.wFree, wHandle: params.wHandle };
    return new ADMMConvexPlanarProjector(faces, x0, p);
  }
  const p: ADMMParams = { rho: params.rho, wFree: params.wFree, wHandle: params.wHandle };
  return new ADMMPlanarProjector(faces, x0, p);
}
