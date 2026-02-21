import type { Plane } from "../math/plane";
import type { Vec3 } from "../math/types";
import { v3 } from "../math/vec3";
import { CONVEX_HALFSPACE_EPS, INVERSE_DENOM_EPS, MIN_RHO } from "../math/constants";
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

function orientPlanesOutward(positions: ReadonlyArray<Vec3>, planes: ReadonlyArray<Plane>): Array<{ n: Vec3; b: number }> {
  let center: Vec3 = [0, 0, 0];
  for (let i = 0; i < positions.length; i++) {
    center = v3.add(center, positions[i]);
  }
  const c = positions.length > 0 ? v3.mul(center, 1 / positions.length) : center;
  return planes.map((plane) => {
    let n: Vec3 = [plane.n[0], plane.n[1], plane.n[2]];
    let b = plane.b;
    const side = v3.dot(n, c) - b;
    if (side > 0) {
      n = v3.mul(n, -1);
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
    memory.z[i] = cloneVec3(seed[i]);
  }
  const oriented = orientPlanesOutward(seed, memory.facePlanes);
  for (let pass = 0; pass < passes; pass++) {
    for (let fi = 0; fi < model.faces.length; fi++) {
      const faceSet = memory.faceVertexSets[fi];
      const { n, b } = oriented[fi];
      const rhs = b - eps;
      for (let vi = 0; vi < memory.z.length; vi++) {
        if (faceSet.has(vi)) continue;
        const d = v3.dot(n, memory.z[vi]) - rhs;
        if (d <= 0) continue;
        memory.z[vi] = v3.sub(memory.z[vi], v3.mul(n, d));
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
    const rho = Math.max(MIN_RHO, params.rho);
    const wFree = Math.max(0, params.wFree);
    const wHandle = Math.max(0, params.wHandle);
    const convexPasses = Math.max(1, model.flavor === "convex" ? Math.floor(params.itersPerFrame) : 1);
    const convexEps = Math.max(0, params.convexHalfspaceEps ?? CONVEX_HALFSPACE_EPS);

    for (let it = 0; it < iterations; it++) {
      for (let vi = 0; vi < state.positions.length; vi++) {
        const deg = memory.incidence[vi].length;
        const target = state.handles.get(vi) ?? state.baseline[vi];
        const w = state.handles.has(vi) ? wHandle : wFree;
        let sum: Vec3 = [0, 0, 0];
        for (let k = 0; k < deg; k++) {
          const { fi, li } = memory.incidence[vi][k];
          const y = memory.y[fi][li];
          const u = memory.u[fi][li];
          sum = v3.add(sum,v3.sub(y,u));
        }
        let denom = w + rho * deg;
        if (model.flavor === "convex") {
          sum = v3.add(sum, v3.sub(memory.z[vi], memory.q[vi]));
          denom += rho;
        }
        const inv = 1 / Math.max(INVERSE_DENOM_EPS, denom);
        state.positions[vi] = v3.mul(v3.add(v3.mul(target, w), v3.mul(sum, rho)), inv);
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
          (p, i) => v3.add(p, memory.q[i])
        );
        projectConvexHalfspaces(model, memory, seed, convexPasses, convexEps);
      }

      updatePlanarDualBlock(model.faces, state.positions, memory.y, memory.u);
      if (model.flavor === "convex") {
        for (let i = 0; i < state.positions.length; i++) {
          memory.q[i] = v3.add(memory.q[i], v3.sub(state.positions[i], memory.z[i]));
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
