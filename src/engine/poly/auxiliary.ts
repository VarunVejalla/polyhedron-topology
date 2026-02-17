import type { Vec3 } from "../math/types";
import type { FaceEdgeIncidence, PlaneEq, PolyAuxState, PolyRichState, PolyState, PolyTopologyData } from "./types";
import { buildPolyTopology } from "./topology";

function dot3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function add3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function mul3(a: ReadonlyArray<number>, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function cross3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalizePlane(plane: PlaneEq): PlaneEq {
  const len = Math.max(1e-12, Math.hypot(plane.n[0], plane.n[1], plane.n[2]));
  const inv = 1 / len;
  return { n: [plane.n[0] * inv, plane.n[1] * inv, plane.n[2] * inv], b: plane.b * inv };
}

function averagePoint(points: ReadonlyArray<Vec3>): Vec3 {
  const c: Vec3 = [0, 0, 0];
  if (points.length === 0) return c;
  for (let i = 0; i < points.length; i++) {
    c[0] += points[i][0];
    c[1] += points[i][1];
    c[2] += points[i][2];
  }
  const inv = 1 / points.length;
  c[0] *= inv;
  c[1] *= inv;
  c[2] *= inv;
  return c;
}

function triangleVolume(a: ReadonlyArray<number>, b: ReadonlyArray<number>, c: ReadonlyArray<number>): number {
  return dot3(a, cross3(b, c)) / 6;
}

function tetraCentroid(a: ReadonlyArray<number>, b: ReadonlyArray<number>, c: ReadonlyArray<number>, d: ReadonlyArray<number>): Vec3 {
  return mul3(add3(add3(a, b), add3(c, d)), 0.25);
}

function computeVolumeAndCenterOfMass(
  state: PolyState,
  options?: { normalizeFacePlanes?: boolean }
): { volume: number; centerOfMass: Vec3 } {
  const normalize = options?.normalizeFacePlanes ?? true;
  const planes = normalize ? state.facePlanes.map(normalizePlane) : state.facePlanes;

  const ref = averagePoint(state.vertices);
  let volume = 0;
  const num: Vec3 = [0, 0, 0];

  for (let fi = 0; fi < state.faces.length; fi++) {
    const face = state.faces[fi];
    if (face.length < 3) continue;
    const n = planes[fi].n;
    const a0 = state.vertices[face[0]];
    for (let i = 1; i + 1 < face.length; i++) {
      let b = state.vertices[face[i]];
      let c = state.vertices[face[i + 1]];
      let triN = cross3(sub3(b, a0), sub3(c, a0));
      if (dot3(triN, n) < 0) {
        const tmp = b;
        b = c;
        c = tmp;
        triN = cross3(sub3(b, a0), sub3(c, a0));
      }
      const pa = sub3(a0, ref);
      const pb = sub3(b, ref);
      const pc = sub3(c, ref);
      let v = triangleVolume(pa, pb, pc);
      if (v < 0) v = -v;
      if (v <= 1e-15) continue;
      const centroid = tetraCentroid(ref, a0, b, c);
      num[0] += centroid[0] * v;
      num[1] += centroid[1] * v;
      num[2] += centroid[2] * v;
      volume += v;
    }
  }

  if (volume <= 1e-15) return { volume: 0, centerOfMass: ref };
  return { volume, centerOfMass: [num[0] / volume, num[1] / volume, num[2] / volume] };
}

function computeEdgeCross(
  vertices: ReadonlyArray<Vec3>,
  topology: PolyTopologyData
): Vec3[] {
  const out: Vec3[] = new Array(topology.edges.length);
  for (let ei = 0; ei < topology.edges.length; ei++) {
    const e = topology.edges[ei];
    const a = vertices[e.a];
    const b = vertices[e.b];
    out[ei] = cross3(a, b);
  }
  return out;
}

function edgeIncidenceArea(
  inc: FaceEdgeIncidence,
  n: ReadonlyArray<number>,
  edgeCross: ReadonlyArray<Vec3>
): number {
  const t = edgeCross[inc.edgeIndex];
  return inc.sign * dot3(n, t);
}

function faceAverageCentroid(state: PolyState, fi: number): Vec3 {
  const face = state.faces[fi];
  if (face.length === 0) return [0, 0, 0];
  const pts: Vec3[] = face.map((vi) => state.vertices[vi]);
  return averagePoint(pts);
}

export function buildPolyAuxState(
  state: PolyState,
  topologyArg?: PolyTopologyData,
  options?: { normalizeFacePlanes?: boolean }
): PolyAuxState {
  const normalize = options?.normalizeFacePlanes ?? true;
  const planes = normalize ? state.facePlanes.map(normalizePlane) : state.facePlanes;
  const topology = topologyArg ?? buildPolyTopology(state.faces, state.vertices.length);

  const edgeCross = computeEdgeCross(state.vertices, topology);

  const faceVectorArea: Vec3[] = new Array(state.faces.length);
  const faceScalarArea: number[] = new Array(state.faces.length);
  const facePyramidVolume: number[] = new Array(state.faces.length);
  const signedIncidenceArea: number[][] = new Array(state.faces.length);
  const faceCentroid: Vec3[] = new Array(state.faces.length);

  for (let fi = 0; fi < state.faces.length; fi++) {
    const n = planes[fi].n;
    const b = planes[fi].b;
    const incs = topology.edgeIncidencesByFace[fi];
    const a: Vec3 = [0, 0, 0];
    const incArea: number[] = new Array(incs.length);
    for (let li = 0; li < incs.length; li++) {
      const inc = incs[li];
      const t = edgeCross[inc.edgeIndex];
      const signedT: Vec3 = [inc.sign * t[0], inc.sign * t[1], inc.sign * t[2]];
      a[0] += 0.5 * signedT[0];
      a[1] += 0.5 * signedT[1];
      a[2] += 0.5 * signedT[2];
      incArea[li] = edgeIncidenceArea(inc, n, edgeCross);
    }
    faceVectorArea[fi] = a;
    const A = dot3(n, a);
    faceScalarArea[fi] = A;
    facePyramidVolume[fi] = (b * A) / 3;
    signedIncidenceArea[fi] = incArea;

    const denom = 6 * A;
    if (Math.abs(denom) <= 1e-12) {
      faceCentroid[fi] = faceAverageCentroid(state, fi);
    } else {
      let sx = 0;
      let sy = 0;
      let sz = 0;
      for (let li = 0; li < incs.length; li++) {
        const inc = incs[li];
        const pI = state.vertices[inc.from];
        const pJ = state.vertices[inc.to];
        const w = incArea[li];
        sx += (pI[0] + pJ[0]) * w;
        sy += (pI[1] + pJ[1]) * w;
        sz += (pI[2] + pJ[2]) * w;
      }
      faceCentroid[fi] = [sx / denom, sy / denom, sz / denom];
    }
  }

  let volume = 0;
  for (let fi = 0; fi < facePyramidVolume.length; fi++) volume += facePyramidVolume[fi];

  let centerOfMass: Vec3 = [0, 0, 0];
  if (Math.abs(volume) > 1e-12) {
    let sx = 0;
    let sy = 0;
    let sz = 0;
    for (let fi = 0; fi < state.faces.length; fi++) {
      const w = facePyramidVolume[fi];
      sx += w * faceCentroid[fi][0];
      sy += w * faceCentroid[fi][1];
      sz += w * faceCentroid[fi][2];
    }
    const scale = 3 / (4 * volume);
    centerOfMass = [sx * scale, sy * scale, sz * scale];
  } else {
    centerOfMass = computeVolumeAndCenterOfMass(state, { normalizeFacePlanes: normalize }).centerOfMass;
  }

  const projectedComByFace: Vec3[] = new Array(state.faces.length);
  const faceComDistance: number[] = new Array(state.faces.length);
  for (let fi = 0; fi < state.faces.length; fi++) {
    const n = planes[fi].n;
    const b = planes[fi].b;
    const d = dot3(n, centerOfMass) - b;
    faceComDistance[fi] = d;
    projectedComByFace[fi] = [
      centerOfMass[0] - d * n[0],
      centerOfMass[1] - d * n[1],
      centerOfMass[2] - d * n[2],
    ];
  }

  return {
    edgeCross,
    faceVectorArea,
    faceScalarArea,
    facePyramidVolume,
    volume,
    faceCentroid,
    centerOfMass,
    projectedComByFace,
    faceComDistance,
    signedIncidenceArea,
  };
}

export function buildPolyRichState(
  state: PolyState,
  topologyArg?: PolyTopologyData,
  options?: { normalizeFacePlanes?: boolean }
): PolyRichState {
  const topology = topologyArg ?? buildPolyTopology(state.faces, state.vertices.length);
  const aux = buildPolyAuxState(state, topology, options);
  return { ...state, topology, aux };
}
