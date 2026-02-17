import type { Plane } from "../geom/plane";
import type { Vec3 } from "../math/types";
import { evaluateQuadratic, quadraticizeAt, zeroQuadratic } from "../optimization/quadratic";
import { OptimizerSession } from "../optimization/session";
import type {
  ConstraintSense,
  FunctionConstraint,
  FunctionConstraintSet,
  OptimizationModel,
  OptimizerKernel,
  QuadraticConstraint,
  QuadraticConstraintSet,
  QuadraticObjectiveProvider,
  QuadraticProvider,
  ScalarFunction,
} from "../optimization/types";
import {
  buildVertexIncidence,
  computePlanarityViolationFromPlanes,
  createPlanarFaceBuffers,
  fitFacePlanesFromPositions,
  updatePlanarDualBlock,
  updatePlanarYBlock,
} from "./admmPlanarShared";
import type { HandleSet, IProjector, ProjectionFlavor, ProjectorParams } from "./index";

type PlanarModel = OptimizationModel & {
  faces: number[][];
  flavor: ProjectionFlavor;
};

type PlanarState = {
  baseline: Vec3[];
  positions: Vec3[];
  handles: Map<number, Vec3>;
};

type PlanarMemory = {
  incidence: Array<Array<{ fi: number; li: number }>>;
  y: Vec3[][];
  u: Vec3[][];
  vbuf: Vec3[][];
  z: Vec3[];
  q: Vec3[];
  vertexNeighbors: number[][];
  faceVertexSets: Array<Set<number>>;
  prevFaceNormals: Array<Vec3 | undefined>;
  facePlanes: Plane[];
  totalPlanarityViolation: number;
};

function makeEmptyQuadraticList(sense: ConstraintSense): QuadraticConstraint[] {
  return sense === "eq" ? [] : [];
}

function makeEmptyFunctionList(sense: ConstraintSense): FunctionConstraint[] {
  return sense === "eq" ? [] : [];
}

const emptyQuadraticConstraints: QuadraticConstraintSet = {
  equalities: makeEmptyQuadraticList("eq"),
  inequalities: makeEmptyQuadraticList("le"),
};

const emptyFunctionConstraints: FunctionConstraintSet = {
  equalities: makeEmptyFunctionList("eq"),
  inequalities: makeEmptyFunctionList("le"),
};

function cloneVec3(p: ReadonlyArray<number>): Vec3 {
  return [p[0], p[1], p[2]];
}

function cloneVec3List(points: ReadonlyArray<ReadonlyArray<number>>): Vec3[] {
  return points.map((p) => cloneVec3(p));
}

function buildVertexNeighbors(faces: ReadonlyArray<ReadonlyArray<number>>, vertexCount: number): number[][] {
  const out = Array.from({ length: vertexCount }, () => new Set<number>());
  for (let fi = 0; fi < faces.length; fi++) {
    const face = faces[fi];
    for (let i = 0; i < face.length; i++) {
      const a = face[i];
      const b = face[(i + 1) % face.length];
      if (a >= 0 && a < vertexCount && b >= 0 && b < vertexCount) {
        out[a].add(b);
        out[b].add(a);
      }
    }
  }
  return out.map((s) => [...s]);
}

function laplacianGradient(positions: ReadonlyArray<Vec3>, neighbors: ReadonlyArray<ReadonlyArray<number>>): Vec3[] {
  const out = positions.map(() => [0, 0, 0] as Vec3);
  for (let vi = 0; vi < positions.length; vi++) {
    const adj = neighbors[vi];
    if (adj.length === 0) continue;
    const avg: Vec3 = [0, 0, 0];
    for (let i = 0; i < adj.length; i++) {
      const p = positions[adj[i]];
      avg[0] += p[0];
      avg[1] += p[1];
      avg[2] += p[2];
    }
    const inv = 1 / adj.length;
    avg[0] *= inv;
    avg[1] *= inv;
    avg[2] *= inv;
    out[vi][0] = positions[vi][0] - avg[0];
    out[vi][1] = positions[vi][1] - avg[1];
    out[vi][2] = positions[vi][2] - avg[2];
  }
  return out;
}

function orientPlanesOutward(positions: ReadonlyArray<Vec3>, planes: ReadonlyArray<Plane>): Array<{ n: Vec3; b: number }> {
  const center: Vec3 = [0, 0, 0];
  for (let i = 0; i < positions.length; i++) {
    center[0] += positions[i][0];
    center[1] += positions[i][1];
    center[2] += positions[i][2];
  }
  if (positions.length > 0) {
    const inv = 1 / positions.length;
    center[0] *= inv;
    center[1] *= inv;
    center[2] *= inv;
  }
  return planes.map((plane) => {
    let n: Vec3 = [plane.n[0], plane.n[1], plane.n[2]];
    let b = plane.b;
    const side = n[0] * center[0] + n[1] * center[1] + n[2] * center[2] - b;
    if (side > 0) {
      n = [-n[0], -n[1], -n[2]];
      b = -b;
    }
    return { n, b };
  });
}

function projectConvexHalfspaces(
  model: PlanarModel,
  memory: PlanarMemory,
  seed: ReadonlyArray<Vec3>,
  passes: number,
  eps: number
): void {
  for (let i = 0; i < memory.z.length; i++) {
    memory.z[i][0] = seed[i][0];
    memory.z[i][1] = seed[i][1];
    memory.z[i][2] = seed[i][2];
  }
  const oriented = orientPlanesOutward(seed, memory.facePlanes);
  for (let pass = 0; pass < passes; pass++) {
    for (let fi = 0; fi < model.faces.length; fi++) {
      const faceSet = memory.faceVertexSets[fi];
      const { n, b } = oriented[fi];
      const rhs = b - eps;
      for (let vi = 0; vi < memory.z.length; vi++) {
        if (faceSet.has(vi)) continue;
        const p = memory.z[vi];
        const d = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - rhs;
        if (d <= 0) continue;
        p[0] -= d * n[0];
        p[1] -= d * n[1];
        p[2] -= d * n[2];
      }
    }
  }
}

function createPlanarMemory(model: PlanarModel, state: PlanarState): PlanarMemory {
  const planar = createPlanarFaceBuffers(model.faces, state.positions);
  const prevFaceNormals = new Array<Vec3 | undefined>(model.faces.length).fill(undefined);
  const facePlanes = fitFacePlanesFromPositions(model.faces, state.positions, prevFaceNormals);
  return {
    incidence: buildVertexIncidence(model.faces, state.positions.length),
    y: planar.y,
    u: planar.u,
    vbuf: planar.vbuf,
    z: cloneVec3List(state.positions),
    q: state.positions.map(() => [0, 0, 0] as Vec3),
    vertexNeighbors: buildVertexNeighbors(model.faces, state.positions.length),
    faceVertexSets: model.faces.map((face) => new Set<number>(face)),
    prevFaceNormals,
    facePlanes,
    totalPlanarityViolation: computePlanarityViolationFromPlanes(model.faces, state.positions, facePlanes),
  };
}

function createZeroFunction(dim: number): ScalarFunction {
  const zero = zeroQuadratic(dim);
  return {
    evaluate: (x) => evaluateQuadratic(zero, x),
    jacobian: () => new Array<number>(dim).fill(0),
    hessian: () => zero.A,
  };
}

const kernel: OptimizerKernel<PlanarState, ProjectorParams, PlanarMemory, PlanarModel> = {
  initialize: ({ model, state }) => createPlanarMemory(model, state),
  step: ({ model, state, params, memory, iterations }) => {
    const rho = Math.max(1e-8, params.rho);
    const wFree = Math.max(0, params.wFree);
    const wHandle = Math.max(0, params.wHandle);
    const lambdaReg = model.flavor === "regular" ? Math.max(0, params.lambdaReg) : 0;
    const convexPasses = Math.max(1, model.flavor === "convex" ? Math.floor(params.itersPerFrame) : 1);
    const convexEps = 1e-6;

    for (let it = 0; it < iterations; it++) {
      const regGrad = lambdaReg > 0 ? laplacianGradient(state.positions, memory.vertexNeighbors) : null;
      for (let vi = 0; vi < state.positions.length; vi++) {
        const deg = memory.incidence[vi].length;
        const target = state.handles.get(vi) ?? state.baseline[vi];
        const w = state.handles.has(vi) ? wHandle : wFree;
        let sum0 = 0;
        let sum1 = 0;
        let sum2 = 0;
        for (let k = 0; k < deg; k++) {
          const { fi, li } = memory.incidence[vi][k];
          const y = memory.y[fi][li];
          const u = memory.u[fi][li];
          sum0 += y[0] - u[0];
          sum1 += y[1] - u[1];
          sum2 += y[2] - u[2];
        }
        let denom = w + rho * deg;
        if (model.flavor === "convex") {
          sum0 += memory.z[vi][0] - memory.q[vi][0];
          sum1 += memory.z[vi][1] - memory.q[vi][1];
          sum2 += memory.z[vi][2] - memory.q[vi][2];
          denom += rho;
        }
        const inv = 1 / Math.max(1e-12, denom);
        state.positions[vi][0] = (w * target[0] + rho * sum0) * inv;
        state.positions[vi][1] = (w * target[1] + rho * sum1) * inv;
        state.positions[vi][2] = (w * target[2] + rho * sum2) * inv;
        if (regGrad) {
          state.positions[vi][0] -= (lambdaReg * regGrad[vi][0]) * inv;
          state.positions[vi][1] -= (lambdaReg * regGrad[vi][1]) * inv;
          state.positions[vi][2] -= (lambdaReg * regGrad[vi][2]) * inv;
        }
      }

      updatePlanarYBlock(
        model.faces,
        state.positions,
        memory.u,
        memory.vbuf,
        memory.y,
        memory.prevFaceNormals,
        memory.facePlanes
      );

      if (model.flavor === "convex") {
        const seed = state.positions.map(
          (p, i) => [p[0] + memory.q[i][0], p[1] + memory.q[i][1], p[2] + memory.q[i][2]] as Vec3
        );
        projectConvexHalfspaces(model, memory, seed, convexPasses, convexEps);
      }

      updatePlanarDualBlock(model.faces, state.positions, memory.y, memory.u);
      if (model.flavor === "convex") {
        for (let i = 0; i < state.positions.length; i++) {
          memory.q[i][0] += state.positions[i][0] - memory.z[i][0];
          memory.q[i][1] += state.positions[i][1] - memory.z[i][1];
          memory.q[i][2] += state.positions[i][2] - memory.z[i][2];
        }
      }
      memory.totalPlanarityViolation = computePlanarityViolationFromPlanes(
        model.faces,
        state.positions,
        memory.facePlanes
      );
    }
  },
};

function buildModel(faces: number[][], flavor: ProjectionFlavor, vertexCount: number): PlanarModel {
  const dim = 3 * vertexCount;
  const zeroFn = createZeroFunction(dim);
  const localConstraints: QuadraticProvider = () => emptyQuadraticConstraints;
  const localObjective: QuadraticObjectiveProvider = (x) => quadraticizeAt(zeroFn, x);
  return {
    quadraticConstraints: emptyQuadraticConstraints,
    exactConstraints: emptyFunctionConstraints,
    localQuadraticConstraints: localConstraints,
    metric: zeroFn,
    regularizer: zeroFn,
    localQuadraticMetric: localObjective,
    localQuadraticRegularizer: localObjective,
    faces,
    flavor,
  };
}

function sanitizeFaces(faces: ReadonlyArray<ReadonlyArray<number>>, vertexCount: number): number[][] {
  const cleaned: number[][] = [];
  for (let fi = 0; fi < faces.length; fi++) {
    const face = faces[fi].filter((vi) => Number.isInteger(vi) && vi >= 0 && vi < vertexCount);
    if (face.length >= 3) cleaned.push([...face]);
  }
  return cleaned;
}

export class PlanarProjector implements IProjector {
  private session: OptimizerSession<PlanarState, ProjectorParams, PlanarMemory, PlanarModel>;
  private readonly faces: number[][];
  private readonly flavor: ProjectionFlavor;

  constructor(facesArg: number[][], x0: Vec3[], flavor: ProjectionFlavor, params: ProjectorParams) {
    this.faces = sanitizeFaces(facesArg, x0.length);
    this.flavor = flavor;
    this.session = new OptimizerSession({
      kernel,
      model: buildModel(this.faces, this.flavor, x0.length),
      initialState: {
        baseline: cloneVec3List(x0),
        positions: cloneVec3List(x0),
        handles: new Map<number, Vec3>(),
      },
      params: { ...params },
    });
  }

  reset(x0: Vec3[]): void {
    const baseline = cloneVec3List(x0);
    this.session.setState({
      baseline,
      positions: cloneVec3List(x0),
      handles: new Map<number, Vec3>(),
    });
  }

  setBaseline(x0: Vec3[]): void {
    const state = this.session.getMutableState();
    state.baseline = cloneVec3List(x0);
  }

  setHandles(handles: HandleSet): void {
    const state = this.session.getMutableState();
    state.handles = new Map<number, Vec3>([...handles.targets.entries()].map(([k, v]) => [k, cloneVec3(v)]));
  }

  setParams(next: Partial<ProjectorParams>): void {
    this.session.setParams(next);
  }

  step(iterations: number): void {
    this.session.step(iterations);
  }

  getPositionsRef(): ReadonlyArray<Vec3> {
    return this.session.getStateRef().positions;
  }

  snapshotPositions(): Vec3[] {
    return cloneVec3List(this.session.getStateRef().positions);
  }

  diagnostics(): { totalPlanarityViolation: number } {
    return { totalPlanarityViolation: this.session.getMemoryRef().totalPlanarityViolation };
  }
}
