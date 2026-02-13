import type { Vec3 } from "../math/types";
import { ADMMPlanarProjector, ADMMParams } from "./planarAdmm";
import { ADMMConvexPlanarProjector, ADMMConvexParams } from "./planarAdmmConvex";
import { ADMMRegularPlanarProjector, ADMMRegularParams } from "./planarAdmmRegular";
import { GuidedALMPlanarProjector, GuidedALMParams } from "./planarGuidedAlm";

export type ProjectionMethod = "admm" | "admm_convex" | "admm_regular" | "guided_alm";

export const projectionMethods: { id: ProjectionMethod; label: string }[] = [
  { id: "admm", label: "ADMM / prox (planar faces)" },
  { id: "admm_convex", label: "ADMM / prox (planar + convex)" },
  { id: "admm_regular", label: "ADMM / prox (planar + face regularity)" },
  { id: "guided_alm", label: "Guided projection + ALM/GN (linearized constraints)" },
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
  setHandles(handles: HandleSet): void;
  step(iterations: number): void;
  getPositionsRef(): ReadonlyArray<Vec3>;
  snapshotPositions(): Vec3[];
  diagnostics(): { totalPlanarityViolation: number };
  // optional runtime param updates
  setParams?(next: ADMMParams | ADMMConvexParams | ADMMRegularParams | GuidedALMParams): void;
}

export function createProjector(method: ProjectionMethod, faces: number[][], x0: Vec3[], params: ProjectorParams): IProjector {
  if (method === "admm_regular") {
    const p: ADMMRegularParams = {
      rho: params.rho,
      wFree: params.wFree,
      wHandle: params.wHandle,
      lambdaReg: params.lambdaReg,
    };
    return new ADMMRegularPlanarProjector(faces, x0, p);
  }
  if (method === "guided_alm") {
    const p: GuidedALMParams = {
      rho: params.rho,
      wFree: params.wFree,
      wHandle: params.wHandle,
      lambdaReg: params.lambdaReg,
    };
    return new GuidedALMPlanarProjector(faces, x0, p);
  }
  if (method === "admm_convex") {
    const p: ADMMConvexParams = { rho: params.rho, wFree: params.wFree, wHandle: params.wHandle };
    return new ADMMConvexPlanarProjector(faces, x0, p);
  }
  const p: ADMMParams = { rho: params.rho, wFree: params.wFree, wHandle: params.wHandle };
  return new ADMMPlanarProjector(faces, x0, p);
}
