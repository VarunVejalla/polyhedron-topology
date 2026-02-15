import type { Vec3 } from "../math/types";
import type { HandleSet, IProjector } from "./index";
import { LinearizedAlmEngine } from "./modular/engines/linearizedAlmEngine";
import {
  computeTotalPlanarityViolation,
  countPlanarGuidedHardConstraints,
  ModularGuidedALMParams,
  orientInitialNormalsOutward,
  packPlanarGuidedY,
  PlanarGuidedModelBuilder,
  unpackPlanarGuidedY,
} from "./modular/planarGuidedModel";
import {
  createArmijoGlobalizer,
  createResidualBalancePenaltyPolicy,
  neverStopPolicy,
  scaledDualUpdater,
} from "./modular/policies";
import { runMetaSolver } from "./modular/solver";
import type { MetaState } from "./modular/types";

export type ModularProjectorParams = ModularGuidedALMParams;

export class ModularPlanarProjector implements IProjector {
  private faces: number[][];
  private x0: Vec3[] = [];
  private x: Vec3[] = [];
  private normals: Vec3[] = [];
  private offsets: number[] = [];
  private params: ModularProjectorParams;
  private handles: HandleSet = { targets: new Map() };
  private lastTotalViolation = 0;

  private state: MetaState = { y: [], u: [], rho: 1 };
  private stepYRef: number[] = [];
  private builder: PlanarGuidedModelBuilder;
  private engine: LinearizedAlmEngine;

  constructor(faces: number[][], x0: Vec3[], params: ModularProjectorParams) {
    this.faces = faces.map((f) => [...f]);
    this.params = { ...params };
    this.engine = new LinearizedAlmEngine({
      cgIters: Math.max(4, Math.floor(this.params.cgIters ?? 80)),
      cgTol: Math.max(1e-10, this.params.cgTol ?? 1e-6),
    });
    this.builder = new PlanarGuidedModelBuilder(
      this.faces,
      x0,
      this.params,
      this.handles,
      () => this.stepYRef
    );
    this.reset(x0);
  }

  reset(x0: Vec3[]) {
    this.x0 = x0.map((p) => [p[0], p[1], p[2]] as Vec3);
    this.x = x0.map((p) => [p[0], p[1], p[2]] as Vec3);

    const oriented = orientInitialNormalsOutward(this.faces, this.x);
    this.normals = oriented.normals;
    this.offsets = oriented.offsets;

    const y = packPlanarGuidedY(this.x, this.normals, this.offsets);
    this.state = {
      y,
      u: new Array<number>(countPlanarGuidedHardConstraints(this.faces)).fill(0),
      rho: this.params.rho,
    };
    this.stepYRef = y.slice();

    this.builder.setBaseline(this.x0);
    this.builder.setParams(this.params);
    this.builder.setHandles(this.handles);

    this.lastTotalViolation = computeTotalPlanarityViolation(this.faces, this.x);
  }

  setHandles(handles: HandleSet) {
    this.handles = handles;
    this.builder.setHandles(handles);
  }

  setParams(next: Partial<ModularProjectorParams>) {
    this.params = { ...this.params, ...next };
    if (next.rho !== undefined) this.state.rho = next.rho;
    this.builder.setParams(next);
  }

  private syncStateToViews() {
    const unpacked = unpackPlanarGuidedY(this.state.y, this.x0.length, this.faces.length);
    this.x = unpacked.vertices;
    this.normals = unpacked.normals;
    this.offsets = unpacked.offsets;
  }

  step(iterations: number) {
    if (iterations <= 0) return;
    if (this.faces.length === 0 || this.x0.length === 0) return;

    this.stepYRef = this.state.y.slice();
    this.builder.setParams(this.params);
    this.builder.setHandles(this.handles);
    this.engine.setParams({
      cgIters: Math.max(4, Math.floor(this.params.cgIters ?? 80)),
      cgTol: Math.max(1e-10, this.params.cgTol ?? 1e-6),
    });

    const lineSearchC1 = this.params.lineSearchC1 ?? 1e-4;
    const lineSearchShrink = Math.min(0.95, Math.max(0.1, this.params.lineSearchShrink ?? 0.5));
    const lineSearchMaxSteps = Math.max(1, Math.floor(this.params.lineSearchMaxSteps ?? 8));
    const minAcceptedAlpha = Math.max(0, Math.min(1, this.params.minAcceptedAlpha ?? 1e-4));

    const globalizer = createArmijoGlobalizer({
      c1: lineSearchC1,
      shrink: lineSearchShrink,
      maxSteps: lineSearchMaxSteps,
    });

    const penaltyPolicy = createResidualBalancePenaltyPolicy({
      enabled: this.params.adaptRho ?? false,
      increase: Math.max(1.01, this.params.rhoIncrease ?? 2),
      decrease: Math.max(1.01, this.params.rhoDecrease ?? 2),
      ratio: Math.max(1.1, this.params.rhoResidualRatio ?? 10),
      min: Math.max(1e-8, this.params.rhoMin ?? 1e-3),
      max: Math.max(Math.max(1e-8, this.params.rhoMin ?? 1e-3), this.params.rhoMax ?? 1e8),
    });

    const stats = runMetaSolver(
      this.state,
      this.builder,
      this.engine,
      globalizer,
      scaledDualUpdater,
      penaltyPolicy,
      neverStopPolicy,
      iterations
    );

    // If line search collapsed this frame, keep previous state (compatibility with current guided_alm behavior).
    if (stats.accepted === 0 || stats.lastAlpha < minAcceptedAlpha) return;

    this.params.rho = this.state.rho;
    this.syncStateToViews();
    this.lastTotalViolation = computeTotalPlanarityViolation(this.faces, this.x);
  }

  getPositionsRef(): ReadonlyArray<Vec3> {
    return this.x;
  }

  snapshotPositions(): Vec3[] {
    return this.x.map((p) => [p[0], p[1], p[2]] as Vec3);
  }

  diagnostics() {
    return { totalPlanarityViolation: this.lastTotalViolation };
  }
}

