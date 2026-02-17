import type { Vec3 } from "../math/types";
import type { PolyAuxState, PolyState, PolyTopologyData } from "./types";
import { buildPolyAuxState } from "./auxiliary";
import { buildPolyTopology } from "./topology";

export type ResidualBlock = {
  name: string;
  values: number[];
  count: number;
  l2: number;
  maxAbs: number;
};

export type PolyConstraintResiduals = {
  edgeCross: ResidualBlock;
  faceVectorArea: ResidualBlock;
  faceScalarArea: ResidualBlock;
  facePyramidVolume: ResidualBlock;
  totalVolume: ResidualBlock;
  incidenceSignedArea: ResidualBlock;
  faceCentroid: ResidualBlock;
  centerOfMass: ResidualBlock;
  projectionDistance: ResidualBlock;
  projectionPlane: ResidualBlock;
  unitNormals: ResidualBlock;
  totalCount: number;
  totalL2: number;
  maxAbs: number;
};

function dot3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function add3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function cross3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function block(name: string, values: number[]): ResidualBlock {
  let s2 = 0;
  let m = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    s2 += v * v;
    const a = Math.abs(v);
    if (a > m) m = a;
  }
  return { name, values, count: values.length, l2: Math.sqrt(s2), maxAbs: m };
}

function normalizePlane(n: Vec3, b: number): { n: Vec3; b: number } {
  const len = Math.max(1e-12, Math.hypot(n[0], n[1], n[2]));
  const inv = 1 / len;
  return { n: [n[0] * inv, n[1] * inv, n[2] * inv], b: b * inv };
}

export function computeAuxConstraintResiduals(
  state: PolyState,
  auxArg?: PolyAuxState,
  topologyArg?: PolyTopologyData
): PolyConstraintResiduals {
  const topology = topologyArg ?? buildPolyTopology(state.faces, state.vertices.length);
  const aux = auxArg ?? buildPolyAuxState(state, topology);

  const edgeCrossRes: number[] = [];
  for (let ei = 0; ei < topology.edges.length; ei++) {
    const e = topology.edges[ei];
    const c = cross3(state.vertices[e.a], state.vertices[e.b]);
    const t = aux.edgeCross[ei];
    edgeCrossRes.push(t[0] - c[0], t[1] - c[1], t[2] - c[2]);
  }

  const faceVectorAreaRes: number[] = [];
  for (let fi = 0; fi < state.faces.length; fi++) {
    const incs = topology.edgeIncidencesByFace[fi];
    const sum: Vec3 = [0, 0, 0];
    for (let li = 0; li < incs.length; li++) {
      const inc = incs[li];
      const t = aux.edgeCross[inc.edgeIndex];
      sum[0] += 0.5 * inc.sign * t[0];
      sum[1] += 0.5 * inc.sign * t[1];
      sum[2] += 0.5 * inc.sign * t[2];
    }
    const a = aux.faceVectorArea[fi];
    faceVectorAreaRes.push(a[0] - sum[0], a[1] - sum[1], a[2] - sum[2]);
  }

  const faceScalarAreaRes: number[] = [];
  const facePyramidVolumeRes: number[] = [];
  const unitNormalsRes: number[] = [];
  for (let fi = 0; fi < state.faces.length; fi++) {
    const pl = normalizePlane(state.facePlanes[fi].n, state.facePlanes[fi].b);
    const A = dot3(pl.n, aux.faceVectorArea[fi]);
    faceScalarAreaRes.push(aux.faceScalarArea[fi] - A);
    facePyramidVolumeRes.push(aux.facePyramidVolume[fi] - (pl.b * aux.faceScalarArea[fi]) / 3);
    unitNormalsRes.push(dot3(pl.n, pl.n) - 1);
  }

  let sumR = 0;
  for (let fi = 0; fi < state.faces.length; fi++) sumR += aux.facePyramidVolume[fi];
  const totalVolumeRes = [aux.volume - sumR];

  const incidenceAreaRes: number[] = [];
  for (let fi = 0; fi < state.faces.length; fi++) {
    const n = normalizePlane(state.facePlanes[fi].n, state.facePlanes[fi].b).n;
    const incs = topology.edgeIncidencesByFace[fi];
    for (let li = 0; li < incs.length; li++) {
      const inc = incs[li];
      const t = aux.edgeCross[inc.edgeIndex];
      const expect = inc.sign * dot3(n, t);
      incidenceAreaRes.push(aux.signedIncidenceArea[fi][li] - expect);
    }
  }

  const centroidRes: number[] = [];
  for (let fi = 0; fi < state.faces.length; fi++) {
    const c = aux.faceCentroid[fi];
    const A = aux.faceScalarArea[fi];
    const lhs = [6 * A * c[0], 6 * A * c[1], 6 * A * c[2]] as Vec3;
    const incs = topology.edgeIncidencesByFace[fi];
    const rhs: Vec3 = [0, 0, 0];
    for (let li = 0; li < incs.length; li++) {
      const inc = incs[li];
      const pI = state.vertices[inc.from];
      const pJ = state.vertices[inc.to];
      const w = aux.signedIncidenceArea[fi][li];
      const s = add3(pI, pJ);
      rhs[0] += s[0] * w;
      rhs[1] += s[1] * w;
      rhs[2] += s[2] * w;
    }
    centroidRes.push(lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]);
  }

  const comRes: number[] = [0, 0, 0];
  const lhsCom = [4 * aux.volume * aux.centerOfMass[0], 4 * aux.volume * aux.centerOfMass[1], 4 * aux.volume * aux.centerOfMass[2]];
  const rhsCom: Vec3 = [0, 0, 0];
  for (let fi = 0; fi < state.faces.length; fi++) {
    rhsCom[0] += 3 * aux.facePyramidVolume[fi] * aux.faceCentroid[fi][0];
    rhsCom[1] += 3 * aux.facePyramidVolume[fi] * aux.faceCentroid[fi][1];
    rhsCom[2] += 3 * aux.facePyramidVolume[fi] * aux.faceCentroid[fi][2];
  }
  comRes[0] = lhsCom[0] - rhsCom[0];
  comRes[1] = lhsCom[1] - rhsCom[1];
  comRes[2] = lhsCom[2] - rhsCom[2];

  const projectionDistanceRes: number[] = [];
  const projectionPlaneRes: number[] = [];
  for (let fi = 0; fi < state.faces.length; fi++) {
    const pl = normalizePlane(state.facePlanes[fi].n, state.facePlanes[fi].b);
    const p = aux.projectedComByFace[fi];
    const c = aux.centerOfMass;
    const d = aux.faceComDistance[fi];
    projectionDistanceRes.push(
      p[0] - c[0] + d * pl.n[0],
      p[1] - c[1] + d * pl.n[1],
      p[2] - c[2] + d * pl.n[2]
    );
    projectionPlaneRes.push(dot3(pl.n, p) - pl.b);
  }

  const edgeCross = block("edgeCross", edgeCrossRes);
  const faceVectorArea = block("faceVectorArea", faceVectorAreaRes);
  const faceScalarArea = block("faceScalarArea", faceScalarAreaRes);
  const facePyramidVolume = block("facePyramidVolume", facePyramidVolumeRes);
  const totalVolume = block("totalVolume", totalVolumeRes);
  const incidenceSignedArea = block("incidenceSignedArea", incidenceAreaRes);
  const faceCentroid = block("faceCentroid", centroidRes);
  const centerOfMass = block("centerOfMass", comRes);
  const projectionDistance = block("projectionDistance", projectionDistanceRes);
  const projectionPlane = block("projectionPlane", projectionPlaneRes);
  const unitNormals = block("unitNormals", unitNormalsRes);

  const blocks = [
    edgeCross,
    faceVectorArea,
    faceScalarArea,
    facePyramidVolume,
    totalVolume,
    incidenceSignedArea,
    faceCentroid,
    centerOfMass,
    projectionDistance,
    projectionPlane,
    unitNormals,
  ];
  let totalCount = 0;
  let totalL2Sq = 0;
  let maxAbs = 0;
  for (let i = 0; i < blocks.length; i++) {
    totalCount += blocks[i].count;
    totalL2Sq += blocks[i].l2 * blocks[i].l2;
    if (blocks[i].maxAbs > maxAbs) maxAbs = blocks[i].maxAbs;
  }

  return {
    edgeCross,
    faceVectorArea,
    faceScalarArea,
    facePyramidVolume,
    totalVolume,
    incidenceSignedArea,
    faceCentroid,
    centerOfMass,
    projectionDistance,
    projectionPlane,
    unitNormals,
    totalCount,
    totalL2: Math.sqrt(totalL2Sq),
    maxAbs,
  };
}

