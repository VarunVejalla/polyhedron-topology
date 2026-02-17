import type { Vec3 } from "../math/types";
import { ADMMPlanarProjector, ADMMParams } from "./planarAdmm";
import { ADMMConvexPlanarProjector, ADMMConvexParams } from "./planarAdmmConvex";
import { ADMMRegularPlanarProjector, ADMMRegularParams } from "./planarAdmmRegular";
import { ModularPlanarProjector, ModularProjectorParams } from "./planarModular";

export type ProjectionMethod =
  | "admm"
  | "admm_convex"
  | "admm_regular"
  | "guided_alm"
  | "guided_alm_squared_slack"
  | "guided_alm_modular";

export const projectionMethods: { id: ProjectionMethod; label: string }[] = [
  { id: "admm", label: "ADMM / prox (planar faces)" },
  { id: "admm_convex", label: "ADMM / prox (planar + convex)" },
  { id: "admm_regular", label: "ADMM / prox (planar + face regularity)" },
  { id: "guided_alm", label: "Guided projection + ALM/GN (linearized constraints)" },
  { id: "guided_alm_squared_slack", label: "Guided projection + ALM (plane vars + squared slacks)" },
  { id: "guided_alm_modular", label: "Guided projection + ALM (modular framework)" },
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
  setParams?(
    next:
      | ADMMParams
      | ADMMConvexParams
      | ADMMRegularParams
      | ModularProjectorParams
  ): void;
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
    const p: ModularProjectorParams = {
      rho: params.rho,
      wFree: params.wFree,
      wHandle: params.wHandle,
      lambdaReg: params.lambdaReg,
      constraintMode: "inc_unit",
    };
    return new ModularPlanarProjector(faces, x0, p);
  }
  if (method === "guided_alm_squared_slack") {
    const p: ModularProjectorParams = {
      rho: params.rho,
      wFree: params.wFree,
      wHandle: params.wHandle,
      lambdaReg: params.lambdaReg,
      constraintMode: "inc_noninc_unit_squared_slack",
    };
    return new ModularPlanarProjector(faces, x0, p);
  }
  if (method === "guided_alm_modular") {
    const p: ModularProjectorParams = {
      rho: params.rho,
      wFree: params.wFree,
      wHandle: params.wHandle,
      lambdaReg: params.lambdaReg,
    };
    return new ModularPlanarProjector(faces, x0, p);
  }
  if (method === "admm_convex") {
    const p: ADMMConvexParams = { rho: params.rho, wFree: params.wFree, wHandle: params.wHandle };
    return new ADMMConvexPlanarProjector(faces, x0, p);
  }
  const p: ADMMParams = { rho: params.rho, wFree: params.wFree, wHandle: params.wHandle };
  return new ADMMPlanarProjector(faces, x0, p);
}
