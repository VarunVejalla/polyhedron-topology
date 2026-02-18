import type { Vec3 } from "../math/types";
import {
  SequentialQuadraticConsensusSolver,
  type SequentialConsensusParams,
} from "../optimization/outer/sequentialQuadraticConsensus";
import { createPolyQcqpLayout, buildHandleMetricQuadratic, cloneVec3List, type ConvexEncoding } from "./polyQcqpModel";
import type { HandleSet, IProjector, ProjectionFlavor, ProjectorParams } from "./index";

export class ConsensusQcqpProjector implements IProjector {
  private readonly flavor: ProjectionFlavor;
  private readonly convexEncoding: ConvexEncoding;
  private readonly layout;
  private baseline: Vec3[];
  private handles: HandleSet = { targets: new Map<number, Vec3>() };
  private params: ProjectorParams;
  private readonly solver: SequentialQuadraticConsensusSolver;
  private positionsCache: Vec3[];

  constructor(
    facesArg: number[][],
    x0: Vec3[],
    flavor: ProjectionFlavor,
    params: ProjectorParams,
    convexEncoding: ConvexEncoding = "slack"
  ) {
    this.flavor = flavor;
    this.convexEncoding = convexEncoding;
    this.layout = createPolyQcqpLayout(facesArg, x0);
    this.baseline = cloneVec3List(x0);
    this.params = { ...params };
    const model = this.layout.buildModel(this.flavor, this.convexEncoding, ({ dim, vertexCount, idxVertex }) =>
      buildHandleMetricQuadratic({
        dim,
        vertexCount,
        idxVertex,
        baseline: this.baseline,
        handles: this.handles.targets,
        wFree: this.params.wFree,
        wHandle: this.params.wHandle,
      })
    );
    const y0 = this.layout.packState(this.baseline, this.flavor, this.convexEncoding);
    this.solver = new SequentialQuadraticConsensusSolver(model, y0, this.solverParams(this.params));
    this.positionsCache = this.layout.unpackVertices(y0);
  }

  private solverParams(params: ProjectorParams): SequentialConsensusParams {
    return {
      rho: Math.max(1e-8, params.rho),
      proximalWeight: 1e-3,
      linearSolveShift: 1e-8,
      qcqpTol: 1e-7,
      qcqpMaxNewtonIters: 20,
      relinearizeEvery: 4,
      innerIterationsPerOuter: 1,
    };
  }

  private syncPositionsFromSolver(): void {
    this.positionsCache = this.layout.unpackVertices(this.solver.getStateRef().x);
  }

  reset(x0: Vec3[]): void {
    this.baseline = cloneVec3List(x0);
    this.handles.targets.clear();
    this.solver.setState(this.layout.packState(this.baseline, this.flavor, this.convexEncoding));
    this.syncPositionsFromSolver();
  }

  setBaseline(x0: Vec3[]): void {
    this.baseline = cloneVec3List(x0);
  }

  setHandles(handles: HandleSet): void {
    this.handles = { targets: new Map(handles.targets) };
  }

  setParams(next: Partial<ProjectorParams>): void {
    this.params = { ...this.params, ...next };
    this.solver.setParams({ rho: this.params.rho });
  }

  step(iterations: number): void {
    this.solver.step(iterations);
    this.syncPositionsFromSolver();
  }

  getPositionsRef(): ReadonlyArray<Vec3> {
    return this.positionsCache;
  }

  snapshotPositions(): Vec3[] {
    return cloneVec3List(this.positionsCache);
  }

  diagnostics(): { totalPlanarityViolation: number } {
    return { totalPlanarityViolation: this.layout.planarityViolation(this.positionsCache) };
  }
}
