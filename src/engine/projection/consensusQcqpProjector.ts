import type { Vec3 } from "../math/types";
import { v3 } from "../math/vec3";
import { SequentialQuadraticConsensusSolver, type SequentialConsensusParams } from "../optimization/outer/sequentialQuadraticConsensus";
import type {
  FunctionConstraintSet,
  IndexedQuadraticConstraint,
  IndexedQuadraticConstraintSet,
  OptimizationModel,
  QuadraticConstraintSet,
  QuadraticForm,
} from "../optimization/types";
import { buildPolyState } from "../poly";
import { buildPolyTopology } from "../poly/topology";
import type { HandleSet, IProjector, ProjectionFlavor, ProjectorParams } from "./index";

const emptyQuadraticConstraints: QuadraticConstraintSet = { equalities: [], inequalities: [] };
const emptyFunctionConstraints: FunctionConstraintSet = { equalities: [], inequalities: [] };

function cloneVec3(p: ReadonlyArray<number>): Vec3 {
  return [p[0], p[1], p[2]];
}

function cloneVec3List(points: ReadonlyArray<ReadonlyArray<number>>): Vec3[] {
  return points.map((p) => cloneVec3(p));
}

function sanitizeFaces(faces: ReadonlyArray<ReadonlyArray<number>>, vertexCount: number): number[][] {
  const cleaned: number[][] = [];
  for (let fi = 0; fi < faces.length; fi++) {
    const face = faces[fi].filter((vi) => Number.isInteger(vi) && vi >= 0 && vi < vertexCount);
    if (face.length >= 3) cleaned.push([...face]);
  }
  return cleaned;
}

function zeroMat(n: number): number[][] {
  return Array.from({ length: n }, () => new Array<number>(n).fill(0));
}

type ConvexEncoding = "slack" | "direct_ineq";

export class ConsensusQcqpProjector implements IProjector {
  private readonly faces: number[][];
  private readonly flavor: ProjectionFlavor;
  private readonly convexEncoding: ConvexEncoding;
  private readonly topology;
  private readonly vertexCount: number;
  private readonly faceCount: number;
  private readonly nonIncidenceCount: number;
  private baseline: Vec3[];
  private handles: HandleSet = { targets: new Map<number, Vec3>() };
  private params: ProjectorParams;
  private readonly model: OptimizationModel;
  private solver: SequentialQuadraticConsensusSolver;
  private positionsCache: Vec3[];

  constructor(
    facesArg: number[][],
    x0: Vec3[],
    flavor: ProjectionFlavor,
    params: ProjectorParams,
    convexEncoding: ConvexEncoding = "slack"
  ) {
    this.vertexCount = x0.length;
    this.faces = sanitizeFaces(facesArg, this.vertexCount);
    this.topology = buildPolyTopology(this.faces, this.vertexCount);
    this.faceCount = this.faces.length;
    this.nonIncidenceCount = this.topology.nonIncidencePairs.length;
    this.flavor = flavor;
    this.convexEncoding = convexEncoding;
    this.baseline = cloneVec3List(x0);
    this.params = { ...params };

    this.model = this.buildModel();
    const y0 = this.packState(this.baseline);
    this.solver = new SequentialQuadraticConsensusSolver(this.model, y0, this.solverParams(this.params));
    this.positionsCache = this.unpackVertices(y0);
  }

  private yDim(): number {
    const base = 3 * this.vertexCount + 4 * this.faceCount;
    if (this.flavor !== "convex") return base;
    return this.convexEncoding === "slack" ? base + this.nonIncidenceCount : base;
  }

  private idxVertex(vi: number): number {
    return 3 * vi;
  }

  private idxFace(fi: number): number {
    return 3 * this.vertexCount + 4 * fi;
  }

  private idxSlack(di: number): number {
    return 3 * this.vertexCount + 4 * this.faceCount + di;
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

  private unpackVertices(y: ReadonlyArray<number>): Vec3[] {
    const out: Vec3[] = new Array(this.vertexCount);
    for (let vi = 0; vi < this.vertexCount; vi++) {
      const b = this.idxVertex(vi);
      out[vi] = [y[b], y[b + 1], y[b + 2]];
    }
    return out;
  }

  private packState(vertices: ReadonlyArray<Vec3>): number[] {
    const y = new Array<number>(this.yDim()).fill(0);
    for (let vi = 0; vi < this.vertexCount; vi++) {
      const b = this.idxVertex(vi);
      y[b] = vertices[vi][0];
      y[b + 1] = vertices[vi][1];
      y[b + 2] = vertices[vi][2];
    }

    const poly = buildPolyState(vertices, this.faces);
    for (let fi = 0; fi < this.faceCount; fi++) {
      const b = this.idxFace(fi);
      const pl = poly.facePlanes[fi];
      y[b] = pl.n[0];
      y[b + 1] = pl.n[1];
      y[b + 2] = pl.n[2];
      y[b + 3] = pl.b;
    }

    if (this.flavor === "convex" && this.convexEncoding === "slack") {
      for (let di = 0; di < this.nonIncidenceCount; di++) {
        const pair = this.topology.nonIncidencePairs[di];
        const fb = this.idxFace(pair.fi);
        const vb = this.idxVertex(pair.vi);
        const gap = y[fb] * y[vb] + y[fb + 1] * y[vb + 1] + y[fb + 2] * y[vb + 2] - y[fb + 3];
        y[this.idxSlack(di)] = Math.sqrt(Math.max(0, -gap));
      }
    }
    return y;
  }

  private incidenceConstraint(fi: number, vi: number): IndexedQuadraticConstraint {
    const fb = this.idxFace(fi);
    const vb = this.idxVertex(vi);
    const indices = [fb, fb + 1, fb + 2, fb + 3, vb, vb + 1, vb + 2];
    const A = zeroMat(7);
    A[0][4] = 1;
    A[4][0] = 1;
    A[1][5] = 1;
    A[5][1] = 1;
    A[2][6] = 1;
    A[6][2] = 1;
    const b = new Array<number>(7).fill(0);
    b[3] = -1;
    return {
      id: `inc:${fi}:${vi}`,
      sense: "eq",
      form: { indices, A, b, c: 0 },
    };
  }

  private unitNormalConstraint(fi: number): IndexedQuadraticConstraint {
    const fb = this.idxFace(fi);
    const indices = [fb, fb + 1, fb + 2];
    const A = zeroMat(3);
    A[0][0] = 2;
    A[1][1] = 2;
    A[2][2] = 2;
    return {
      id: `unit:${fi}`,
      sense: "eq",
      form: { indices, A, b: [0, 0, 0], c: -1 },
    };
  }

  private nonIncSlackConstraint(fi: number, vi: number, di: number): IndexedQuadraticConstraint {
    const fb = this.idxFace(fi);
    const vb = this.idxVertex(vi);
    const sb = this.idxSlack(di);
    const indices = [fb, fb + 1, fb + 2, fb + 3, vb, vb + 1, vb + 2, sb];
    const A = zeroMat(8);
    A[0][4] = 1;
    A[4][0] = 1;
    A[1][5] = 1;
    A[5][1] = 1;
    A[2][6] = 1;
    A[6][2] = 1;
    A[7][7] = 2;
    const b = new Array<number>(8).fill(0);
    b[3] = -1;
    return {
      id: `noninc:${fi}:${vi}:${di}`,
      sense: "eq",
      form: { indices, A, b, c: 0 },
    };
  }

  private nonIncIneqConstraint(fi: number, vi: number): IndexedQuadraticConstraint {
    const fb = this.idxFace(fi);
    const vb = this.idxVertex(vi);
    const indices = [fb, fb + 1, fb + 2, fb + 3, vb, vb + 1, vb + 2];
    const A = zeroMat(7);
    A[0][4] = 1;
    A[4][0] = 1;
    A[1][5] = 1;
    A[5][1] = 1;
    A[2][6] = 1;
    A[6][2] = 1;
    const b = new Array<number>(7).fill(0);
    b[3] = -1;
    return {
      id: `noninc_ineq:${fi}:${vi}`,
      sense: "le",
      form: { indices, A, b, c: 0 },
    };
  }

  private metricQuadraticForm(): QuadraticForm {
    const dim = this.yDim();
    const A = zeroMat(dim);
    const b = new Array<number>(dim).fill(0);
    let c = 0;
    for (let vi = 0; vi < this.vertexCount; vi++) {
      const base = this.idxVertex(vi);
      const target = this.handles.targets.get(vi) ?? this.baseline[vi];
      const w = this.handles.targets.has(vi) ? this.params.wHandle : this.params.wFree;
      const s = 2 * w;
      A[base][base] = s;
      A[base + 1][base + 1] = s;
      A[base + 2][base + 2] = s;
      b[base] = -s * target[0];
      b[base + 1] = -s * target[1];
      b[base + 2] = -s * target[2];
      c += w * (target[0] * target[0] + target[1] * target[1] + target[2] * target[2]);
    }
    for (let i = 3 * this.vertexCount; i < dim; i++) {
      A[i][i] += 2e-6;
    }
    return { A, b, c };
  }

  private buildModel(): OptimizationModel {
    const indexed: IndexedQuadraticConstraintSet = { equalities: [], inequalities: [] };

    for (let i = 0; i < this.topology.incidencePairs.length; i++) {
      const pair = this.topology.incidencePairs[i];
      indexed.equalities.push(this.incidenceConstraint(pair.fi, pair.vi));
    }
    for (let fi = 0; fi < this.faceCount; fi++) indexed.equalities.push(this.unitNormalConstraint(fi));
    if (this.flavor === "convex") {
      if (this.convexEncoding === "slack") {
        for (let di = 0; di < this.nonIncidenceCount; di++) {
          const pair = this.topology.nonIncidencePairs[di];
          indexed.equalities.push(this.nonIncSlackConstraint(pair.fi, pair.vi, di));
        }
      } else {
        for (let di = 0; di < this.nonIncidenceCount; di++) {
          const pair = this.topology.nonIncidencePairs[di];
          indexed.inequalities.push(this.nonIncIneqConstraint(pair.fi, pair.vi));
        }
      }
    }

    return {
      quadraticConstraints: emptyQuadraticConstraints,
      indexedQuadraticConstraints: indexed,
      exactConstraints: emptyFunctionConstraints,
      localQuadraticMetric: () => this.metricQuadraticForm(),
    };
  }

  private syncPositionsFromSolver(): void {
    this.positionsCache = this.unpackVertices(this.solver.getStateRef().x);
  }

  reset(x0: Vec3[]): void {
    this.baseline = cloneVec3List(x0);
    const y0 = this.packState(this.baseline);
    this.handles.targets.clear();
    this.solver.setState(y0);
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
    const poly = buildPolyState(this.positionsCache, this.faces);
    let total = 0;
    for (let fi = 0; fi < this.faces.length; fi++) {
      const face = this.faces[fi];
      const pl = poly.facePlanes[fi];
      for (let i = 0; i < face.length; i++) total += Math.abs(v3.dot(pl.n, this.positionsCache[face[i]]) - pl.b);
    }
    return { totalPlanarityViolation: total };
  }
}
