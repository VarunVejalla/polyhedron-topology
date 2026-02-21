import type { Vec3 } from "../math/types";
import { v3 } from "../math/vec3";
import { PLANE_VARIABLE_REGULARIZATION } from "../math/constants";
import { buildPolyState } from "../poly";
import { buildPolyTopology } from "../poly/topology";
import type {
  FunctionConstraintSet,
  IndexedQuadraticConstraint,
  IndexedQuadraticConstraintSet,
  OptimizationModel,
  QuadraticConstraintSet,
  QuadraticForm,
} from "../optimization/types";
import type { ProjectionFlavor } from "./index";

export type ConvexEncoding = "slack" | "direct_ineq";

const emptyQuadraticConstraints: QuadraticConstraintSet = { equalities: [], inequalities: [] };
const emptyFunctionConstraints: FunctionConstraintSet = { equalities: [], inequalities: [] };

function cloneVec3(p: ReadonlyArray<number>): Vec3 {
  return [p[0], p[1], p[2]];
}

export function cloneVec3List(points: ReadonlyArray<ReadonlyArray<number>>): Vec3[] {
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

type MetricBuilder = (args: {
  dim: number;
  vertexCount: number;
  idxVertex: (vi: number) => number;
}) => QuadraticForm;

type PolyQcqpBuildOptions = {
  includeNondegeneracy?: boolean;
};

type PolyQcqpLayout = {
  faces: number[][];
  topology: ReturnType<typeof buildPolyTopology>;
  vertexCount: number;
  faceCount: number;
  nonIncidenceCount: number;
  yDim: (flavor: ProjectionFlavor, convexEncoding: ConvexEncoding) => number;
  idxVertex: (vi: number) => number;
  idxFace: (fi: number) => number;
  idxSlack: (di: number, flavor: ProjectionFlavor, convexEncoding: ConvexEncoding) => number;
  packState: (vertices: ReadonlyArray<Vec3>, flavor: ProjectionFlavor, convexEncoding: ConvexEncoding) => number[];
  unpackVertices: (y: ReadonlyArray<number>) => Vec3[];
  buildModel: (
    flavor: ProjectionFlavor,
    convexEncoding: ConvexEncoding,
    metricBuilder: MetricBuilder,
    options?: PolyQcqpBuildOptions
  ) => OptimizationModel;
  planarityViolation: (vertices: ReadonlyArray<Vec3>) => number;
};

function incidenceConstraint(layout: PolyQcqpLayout, fi: number, vi: number): IndexedQuadraticConstraint {
  const fb = layout.idxFace(fi);
  const vb = layout.idxVertex(vi);
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
  return { id: `inc:${fi}:${vi}`, sense: "eq", form: { indices, A, b, c: 0 } };
}

function unitNormalConstraint(layout: PolyQcqpLayout, fi: number): IndexedQuadraticConstraint {
  const fb = layout.idxFace(fi);
  const indices = [fb, fb + 1, fb + 2];
  const A = zeroMat(3);
  A[0][0] = 2;
  A[1][1] = 2;
  A[2][2] = 2;
  return { id: `unit:${fi}`, sense: "eq", form: { indices, A, b: [0, 0, 0], c: -1 } };
}

function nonIncSlackConstraint(layout: PolyQcqpLayout, fi: number, vi: number, di: number): IndexedQuadraticConstraint {
  const fb = layout.idxFace(fi);
  const vb = layout.idxVertex(vi);
  const sb = layout.idxSlack(di, "convex", "slack");
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
  return { id: `noninc:${fi}:${vi}:${di}`, sense: "eq", form: { indices, A, b, c: 0 } };
}

function nonIncIneqConstraint(layout: PolyQcqpLayout, fi: number, vi: number): IndexedQuadraticConstraint {
  const fb = layout.idxFace(fi);
  const vb = layout.idxVertex(vi);
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
  return { id: `noninc_ineq:${fi}:${vi}`, sense: "le", form: { indices, A, b, c: 0 } };
}

function meanCoordinateConstraint(layout: PolyQcqpLayout, axis: 0 | 1 | 2): IndexedQuadraticConstraint {
  const indices = new Array<number>(layout.vertexCount);
  const b = new Array<number>(layout.vertexCount).fill(1);
  for (let vi = 0; vi < layout.vertexCount; vi++) indices[vi] = layout.idxVertex(vi) + axis;
  return {
    id: `mean:${axis}`,
    sense: "eq",
    form: {
      indices,
      A: zeroMat(layout.vertexCount),
      b,
      c: 0,
    },
  };
}

function vertexNormConstraint(layout: PolyQcqpLayout): IndexedQuadraticConstraint {
  const indices = new Array<number>(3 * layout.vertexCount);
  for (let vi = 0; vi < layout.vertexCount; vi++) {
    const b = layout.idxVertex(vi);
    indices[3 * vi] = b;
    indices[3 * vi + 1] = b + 1;
    indices[3 * vi + 2] = b + 2;
  }
  const A = zeroMat(indices.length);
  for (let i = 0; i < indices.length; i++) A[i][i] = 2;
  return {
    id: "vertex_norm",
    sense: "eq",
    form: {
      indices,
      A,
      b: new Array<number>(indices.length).fill(0),
      c: -layout.vertexCount,
    },
  };
}

export function createPolyQcqpLayout(facesArg: number[][], x0: ReadonlyArray<Vec3>): PolyQcqpLayout {
  const vertexCount = x0.length;
  const faces = sanitizeFaces(facesArg, vertexCount);
  const topology = buildPolyTopology(faces, vertexCount);
  const faceCount = faces.length;
  const nonIncidenceCount = topology.nonIncidencePairs.length;

  const yDim = (flavor: ProjectionFlavor, convexEncoding: ConvexEncoding): number => {
    const base = 3 * vertexCount + 4 * faceCount;
    if (flavor !== "convex") return base;
    return convexEncoding === "slack" ? base + nonIncidenceCount : base;
  };

  const idxVertex = (vi: number): number => 3 * vi;
  const idxFace = (fi: number): number => 3 * vertexCount + 4 * fi;
  const idxSlack = (di: number, flavor: ProjectionFlavor, convexEncoding: ConvexEncoding): number => {
    if (flavor !== "convex" || convexEncoding !== "slack") return -1;
    return 3 * vertexCount + 4 * faceCount + di;
  };

  const unpackVertices = (y: ReadonlyArray<number>): Vec3[] => {
    const out: Vec3[] = new Array(vertexCount);
    for (let vi = 0; vi < vertexCount; vi++) {
      const b = idxVertex(vi);
      out[vi] = [y[b], y[b + 1], y[b + 2]];
    }
    return out;
  };

  const packState = (vertices: ReadonlyArray<Vec3>, flavor: ProjectionFlavor, convexEncoding: ConvexEncoding): number[] => {
    const y = new Array<number>(yDim(flavor, convexEncoding)).fill(0);
    for (let vi = 0; vi < vertexCount; vi++) {
      const b = idxVertex(vi);
      y[b] = vertices[vi][0];
      y[b + 1] = vertices[vi][1];
      y[b + 2] = vertices[vi][2];
    }

    const poly = buildPolyState(vertices, faces);
    for (let fi = 0; fi < faceCount; fi++) {
      const b = idxFace(fi);
      const pl = poly.facePlanes[fi];
      y[b] = pl.n[0];
      y[b + 1] = pl.n[1];
      y[b + 2] = pl.n[2];
      y[b + 3] = pl.b;
    }

    if (flavor === "convex" && convexEncoding === "slack") {
      for (let di = 0; di < nonIncidenceCount; di++) {
        const pair = topology.nonIncidencePairs[di];
        const fb = idxFace(pair.fi);
        const vb = idxVertex(pair.vi);
        const gap = y[fb] * y[vb] + y[fb + 1] * y[vb + 1] + y[fb + 2] * y[vb + 2] - y[fb + 3];
        y[idxSlack(di, flavor, convexEncoding)] = Math.sqrt(Math.max(0, -gap));
      }
    }
    return y;
  };

  const buildModel = (
    flavor: ProjectionFlavor,
    convexEncoding: ConvexEncoding,
    metricBuilder: MetricBuilder,
    options?: PolyQcqpBuildOptions
  ): OptimizationModel => {
    const includeNondegeneracy = options?.includeNondegeneracy ?? true;
    const indexed: IndexedQuadraticConstraintSet = { equalities: [], inequalities: [] };
    for (let i = 0; i < topology.incidencePairs.length; i++) {
      const pair = topology.incidencePairs[i];
      indexed.equalities.push(incidenceConstraint(layout, pair.fi, pair.vi));
    }
    for (let fi = 0; fi < faceCount; fi++) indexed.equalities.push(unitNormalConstraint(layout, fi));
    if (includeNondegeneracy) {
      indexed.equalities.push(meanCoordinateConstraint(layout, 0));
      indexed.equalities.push(meanCoordinateConstraint(layout, 1));
      indexed.equalities.push(meanCoordinateConstraint(layout, 2));
      indexed.equalities.push(vertexNormConstraint(layout));
    }

    if (flavor === "convex") {
      if (convexEncoding === "slack") {
        for (let di = 0; di < nonIncidenceCount; di++) {
          const pair = topology.nonIncidencePairs[di];
          indexed.equalities.push(nonIncSlackConstraint(layout, pair.fi, pair.vi, di));
        }
      } else {
        for (let di = 0; di < nonIncidenceCount; di++) {
          const pair = topology.nonIncidencePairs[di];
          indexed.inequalities.push(nonIncIneqConstraint(layout, pair.fi, pair.vi));
        }
      }
    }

    return {
      quadraticConstraints: emptyQuadraticConstraints,
      indexedQuadraticConstraints: indexed,
      exactConstraints: emptyFunctionConstraints,
      localQuadraticMetric: () => metricBuilder({
        dim: yDim(flavor, convexEncoding),
        vertexCount,
        idxVertex,
      }),
    };
  };

  const planarityViolation = (vertices: ReadonlyArray<Vec3>): number => {
    const poly = buildPolyState(vertices, faces);
    let total = 0;
    for (let fi = 0; fi < faces.length; fi++) {
      const face = faces[fi];
      const pl = poly.facePlanes[fi];
      for (let i = 0; i < face.length; i++) total += Math.abs(v3.dot(pl.n, vertices[face[i]]) - pl.b);
    }
    return total;
  };

  const layout: PolyQcqpLayout = {
    faces,
    topology,
    vertexCount,
    faceCount,
    nonIncidenceCount,
    yDim,
    idxVertex,
    idxFace,
    idxSlack,
    packState,
    unpackVertices,
    buildModel,
    planarityViolation,
  };
  return layout;
}

export function buildHandleMetricQuadratic(args: {
  dim: number;
  vertexCount: number;
  idxVertex: (vi: number) => number;
  baseline: ReadonlyArray<Vec3>;
  handles: ReadonlyMap<number, Vec3>;
  wFree: number;
  wHandle: number;
}): QuadraticForm {
  const A = zeroMat(args.dim);
  const b = new Array<number>(args.dim).fill(0);
  let c = 0;

  for (let vi = 0; vi < args.vertexCount; vi++) {
    const base = args.idxVertex(vi);
    const target = args.handles.get(vi) ?? args.baseline[vi];
    const w = args.handles.has(vi) ? args.wHandle : args.wFree;
    const s = 2 * w;
    A[base][base] = s;
    A[base + 1][base + 1] = s;
    A[base + 2][base + 2] = s;
    b[base] = -s * target[0];
    b[base + 1] = -s * target[1];
    b[base + 2] = -s * target[2];
    c += w * (target[0] * target[0] + target[1] * target[1] + target[2] * target[2]);
  }

  for (let i = 3 * args.vertexCount; i < args.dim; i++) A[i][i] += PLANE_VARIABLE_REGULARIZATION;
  return { A, b, c };
}
