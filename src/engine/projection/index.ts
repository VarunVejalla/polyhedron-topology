import type { Vec3 } from "../math/types";
import { PlanarProjector } from "./planarOptimizer";
import { GuidedAlmProjector } from "./guidedAlmProjector";
import { GuidedAlmLegacyProjector } from "./guidedAlmLegacyProjector";
import { ConsensusQcqpProjector } from "./consensusQcqpProjector";

export type ProjectionMethod =
  | "planar"
  | "guided_alm"
  | "guided_alm_legacy"
  | "consensus_qcqp";

export type ProjectionFlavor = "planar" | "convex";

export const projectionMethods: { id: ProjectionMethod; label: string }[] = [
  { id: "planar", label: "Planar ADMM" },
  { id: "guided_alm", label: "Guided ALM (quadratic)" },
  { id: "guided_alm_legacy", label: "Guided ALM Legacy" },
  { id: "consensus_qcqp", label: "Consensus QCQP" },
];

export type HandleSet = {
  targets: Map<number, Vec3>;
};

export type ProjectorParams = {
  rho: number;
  wFree: number;
  wHandle: number;
  useConvexConstraint?: boolean;
  useVolumeConstraint?: boolean;
  goalVolume?: number;
  itersPerFrame: number;
  itersOnRelease: number;
  qcqpDamping?: number;
  almProximalWeight?: number;
  almActiveSetEps?: number;
  almMaxStepNorm?: number;
  almMinStepScale?: number;
  almMaxBacktracks?: number;
  almDualRelaxation?: number;
  almLambdaClip?: number;
  convexHalfspaceEps?: number;
  legacyStepCapRatio?: number;
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
  const flavor: ProjectionFlavor = params.useConvexConstraint ? "convex" : "planar";
  if (method === "guided_alm") return new GuidedAlmProjector(faces, x0, flavor, params);
  if (method === "guided_alm_legacy") return new GuidedAlmLegacyProjector(faces, x0, flavor, params);
  if (method === "consensus_qcqp") return new ConsensusQcqpProjector(faces, x0, flavor, params);
  return new PlanarProjector(faces, x0, flavor, params);
}
