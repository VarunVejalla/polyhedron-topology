import type { Vec3 } from "../math/types";
import { ConsensusQcqpSolver } from "../optimization/consensusQcqpSolver";
import type { OptimizationModel } from "../optimization/types";
import {
  buildHandleMetricQuadratic,
  cloneVec3List,
  createPolyQcqpLayout,
  type ConvexEncoding,
} from "./polyQcqpModel";
import type { HandleSet, IProjector, ProjectionFlavor, ProjectorParams } from "./index";

function cloneVec3(p: ReadonlyArray<number>): Vec3 {
  return [p[0], p[1], p[2]];
}

function cloneHandleMap(map: ReadonlyMap<number, Vec3>): Map<number, Vec3> {
  return new Map<number, Vec3>([...map.entries()].map(([k, v]) => [k, cloneVec3(v)]));
}

export class ConsensusQcqpProjector implements IProjector {
  private readonly flavor: ProjectionFlavor;
  private readonly convexEncoding: ConvexEncoding;
  private readonly layout: ReturnType<typeof createPolyQcqpLayout>;
  private readonly model: OptimizationModel;
  private readonly dim: number;
  private baseline: Vec3[];
  private positions: Vec3[];
  private handles = new Map<number, Vec3>();
  private params: ProjectorParams;
  private solver: ConsensusQcqpSolver;

  constructor(faces: number[][], x0: Vec3[], flavor: ProjectionFlavor, params: ProjectorParams) {
    this.flavor = flavor;
    this.convexEncoding = "direct_ineq";
    this.layout = createPolyQcqpLayout(faces, x0);
    this.dim = this.layout.yDim(this.flavor, this.convexEncoding);
    this.baseline = cloneVec3List(x0);
    this.positions = cloneVec3List(x0);
    this.params = { ...params };
    this.model = this.layout.buildModel(
      this.flavor,
      this.convexEncoding,
      ({ dim, vertexCount, idxVertex }) =>
        buildHandleMetricQuadratic({
          dim,
          vertexCount,
          idxVertex,
          baseline: this.baseline,
          handles: this.handles,
          wFree: Math.max(0, this.params.wFree),
          wHandle: Math.max(0, this.params.wHandle),
        }),
      { includeNondegeneracy: false }
    );
    this.solver = new ConsensusQcqpSolver({
      model: this.model,
      initialX: this.layout.packState(this.positions, this.flavor, this.convexEncoding),
      rho: params.rho,
      damping: 1e-6,
    });
  }

  reset(x0: Vec3[]): void {
    this.baseline = cloneVec3List(x0);
    this.positions = cloneVec3List(x0);
    this.handles.clear();
    this.solver.resetState(this.layout.packState(this.positions, this.flavor, this.convexEncoding));
  }

  setBaseline(x0: Vec3[]): void {
    this.baseline = cloneVec3List(x0);
  }

  setHandles(handles: HandleSet): void {
    this.handles = cloneHandleMap(handles.targets);
  }

  setParams(next: Partial<ProjectorParams>): void {
    this.params = { ...this.params, ...next };
    this.solver.setParams({ rho: Math.max(1e-8, this.params.rho) });
  }

  step(iterations: number): void {
    if (iterations <= 0) return;
    this.solver.step(iterations);
    const y = this.solver.getStateRef();
    if (y.length !== this.dim) return;
    this.positions = this.layout.unpackVertices(y);
  }

  getPositionsRef(): ReadonlyArray<Vec3> {
    return this.positions;
  }

  snapshotPositions(): Vec3[] {
    return cloneVec3List(this.positions);
  }

  diagnostics(): { totalPlanarityViolation: number } {
    return { totalPlanarityViolation: this.layout.planarityViolation(this.positions) };
  }
}
