import type { Vec3 } from "../math/types";
import { PlanarProjector } from "./planarOptimizer";
import { GuidedAlmProjector } from "./guidedAlmProjector";
import { GuidedAlmLegacyProjector } from "./guidedAlmLegacyProjector";
import { ConsensusQcqpProjector } from "./consensusQcqpProjector";

export type ProjectionMethod =
  | "planar"
  | "convex"
  | "guided_alm_planar"
  | "guided_alm_convex"
  | "guided_alm_legacy_planar"
  | "guided_alm_legacy_convex"
  | "consensus_qcqp_planar"
  | "consensus_qcqp_convex_direct";

export type ProjectionFlavor = "planar" | "convex";

export const projectionMethods: { id: ProjectionMethod; label: string }[] = [
  { id: "planar", label: "Planar projection" },
  { id: "convex", label: "Planar + convex projection" },
  { id: "guided_alm_planar", label: "Guided ALM (quadratic planar)" },
  { id: "guided_alm_convex", label: "Guided ALM (quadratic convex direct)" },
  { id: "guided_alm_legacy_planar", label: "Guided ALM Legacy (planar)" },
  { id: "guided_alm_legacy_convex", label: "Guided ALM Legacy (convex)" },
  { id: "consensus_qcqp_planar", label: "Consensus QCQP (planar)" },
  { id: "consensus_qcqp_convex_direct", label: "Consensus QCQP (convex direct)" },
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
  if (method === "guided_alm_planar") return new GuidedAlmProjector(faces, x0, "planar", params);
  if (method === "guided_alm_convex") return new GuidedAlmProjector(faces, x0, "convex", params);
  if (method === "guided_alm_legacy_planar") return new GuidedAlmLegacyProjector(faces, x0, "planar", params);
  if (method === "guided_alm_legacy_convex") return new GuidedAlmLegacyProjector(faces, x0, "convex", params);
  if (method === "consensus_qcqp_planar") return new ConsensusQcqpProjector(faces, x0, "planar", params);
  if (method === "consensus_qcqp_convex_direct") return new ConsensusQcqpProjector(faces, x0, "convex", params);
  return new PlanarProjector(faces, x0, method === "convex" ? "convex" : "planar", params);
}
