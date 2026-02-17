import type { Vec3 } from "../math/types";
import { PiecewiseOptimizationController } from "../optimization/controller";
import { LinearizedAlmKernel } from "../optimization/kernels/linearizedAlmKernel";
import type { OptimizerHyperParams } from "../optimization/types";
import type { HandleSet, IProjector } from "./index";
import {
  computeTotalPlanarityViolation,
  type ModularConstraintMode,
  ModularGuidedALMParams,
  orientInitialNormalsOutward,
  packPlanarGuidedY,
  PlanarGuidedProblemBuilder,
  unpackPlanarGuidedY,
} from "./modular/planarGuidedModel";

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
  private stepYRef: number[] = [];
  private builder: PlanarGuidedProblemBuilder;
  private kernel: LinearizedAlmKernel;
  private controller!: PiecewiseOptimizationController<void>;

  private mode(): ModularConstraintMode {
    return this.params.constraintMode ?? "inc_unit";
  }

  constructor(faces: number[][], x0: Vec3[], params: ModularProjectorParams) {
    this.faces = faces.map((f) => [...f]);
    this.params = { ...params };
    this.kernel = new LinearizedAlmKernel();
    this.builder = new PlanarGuidedProblemBuilder(
      this.faces,
      x0,
      this.params,
      this.handles,
      () => this.stepYRef,
      this.mode()
    );
    this.reset(x0);
  }

  private hyperParams(): OptimizerHyperParams {
    return {
      rho: this.params.rho,
      tau: this.params.tau ?? 1e-6,
      cgIters: this.params.cgIters ?? 80,
      cgTol: this.params.cgTol ?? 1e-6,
      lineSearchC1: this.params.lineSearchC1 ?? 1e-4,
      lineSearchShrink: this.params.lineSearchShrink ?? 0.5,
      lineSearchMaxSteps: this.params.lineSearchMaxSteps ?? 8,
      adaptRho: this.params.adaptRho ?? false,
      rhoIncrease: this.params.rhoIncrease ?? 2,
      rhoDecrease: this.params.rhoDecrease ?? 2,
      rhoResidualRatio: this.params.rhoResidualRatio ?? 10,
      rhoMin: this.params.rhoMin ?? 1e-3,
      rhoMax: this.params.rhoMax ?? 1e8,
    };
  }

  reset(x0: Vec3[]) {
    this.x0 = x0.map((p) => [p[0], p[1], p[2]] as Vec3);
    this.x = x0.map((p) => [p[0], p[1], p[2]] as Vec3);

    const oriented = orientInitialNormalsOutward(this.faces, this.x);
    this.normals = oriented.normals;
    this.offsets = oriented.offsets;

    const y = packPlanarGuidedY(this.x, this.normals, this.offsets, this.mode(), this.faces);
    this.stepYRef = y.slice();

    this.builder.setBaseline(this.x0);
    this.builder.setParams(this.params);
    this.builder.setHandles(this.handles);
    this.controller = new PiecewiseOptimizationController<void>({
      builder: this.builder,
      kernel: this.kernel,
      x0: y,
      hyperParams: this.hyperParams(),
    });

    const unpacked = unpackPlanarGuidedY(y, this.x0.length, this.faces.length, this.mode(), this.faces);
    this.x = unpacked.vertices;
    this.normals = unpacked.normals;
    this.offsets = unpacked.offsets;
    this.lastTotalViolation = computeTotalPlanarityViolation(this.faces, this.x);
  }

  setHandles(handles: HandleSet) {
    this.handles = handles;
    this.builder.setHandles(handles);
  }

  setParams(next: Partial<ModularProjectorParams>) {
    const prevMode = this.mode();
    this.params = { ...this.params, ...next };
    const nextMode = this.mode();
    if (prevMode !== nextMode) {
      this.builder = new PlanarGuidedProblemBuilder(
        this.faces,
        this.x0,
        this.params,
        this.handles,
        () => this.stepYRef,
        nextMode
      );
      this.reset(this.x0);
      return;
    }
    this.builder.setParams(next);
    this.controller.setHyperParams(this.hyperParams());
  }

  private syncStateToViews() {
    const y = this.controller.getXRef();
    const unpacked = unpackPlanarGuidedY(y, this.x0.length, this.faces.length, this.mode(), this.faces);
    this.x = unpacked.vertices;
    this.normals = unpacked.normals;
    this.offsets = unpacked.offsets;
  }

  step(iterations: number) {
    if (iterations <= 0) return;
    if (this.faces.length === 0 || this.x0.length === 0) return;

    this.stepYRef = [...this.controller.getXRef()];
    this.builder.setParams(this.params);
    this.builder.setHandles(this.handles);
    this.controller.setHyperParams(this.hyperParams());

    const minAcceptedAlpha = Math.max(0, Math.min(1, this.params.minAcceptedAlpha ?? 1e-4));
    const stats = this.controller.step(iterations);
    if (stats.accepted === 0 || stats.lastAlpha < minAcceptedAlpha) return;

    this.params.rho = this.controller.getMutableState().rho;
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
