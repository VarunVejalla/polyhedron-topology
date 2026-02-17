import type { Vec3 } from "../math/types";
import type { FaceEdgeIncidence, PlaneEq, PolyAuxState, PolyRichState, PolyState, PolyTopologyData } from "./types";
import { buildPolyTopology } from "./topology";

function dot3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
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

function averageVertices(vertices: ReadonlyArray<Vec3>, ids?: ReadonlyArray<number>): Vec3 {
  const c: Vec3 = [0, 0, 0];
  const n = ids ? ids.length : vertices.length;
  if (n === 0) return c;
  if (!ids) {
    for (let i = 0; i < vertices.length; i++) {
      c[0] += vertices[i][0];
      c[1] += vertices[i][1];
      c[2] += vertices[i][2];
    }
  } else {
    for (let i = 0; i < ids.length; i++) {
      const p = vertices[ids[i]];
      c[0] += p[0];
      c[1] += p[1];
      c[2] += p[2];
    }
  }
  const inv = 1 / n;
  return [c[0] * inv, c[1] * inv, c[2] * inv];
}

export function buildPolyAuxState(
  state: PolyState,
  topologyArg?: PolyTopologyData,
  options?: { normalizeFacePlanes?: boolean }
): PolyAuxState {
  const normalize = options?.normalizeFacePlanes ?? true;
  const planes = normalize ? state.facePlanes.map(normalizePlane) : state.facePlanes;
  const topology = topologyArg ?? buildPolyTopology(state.faces, state.vertices.length);

  const edgeCross: Vec3[] = new Array(topology.edges.length);
  for (let ei = 0; ei < topology.edges.length; ei++) {
    const e = topology.edges[ei];
    edgeCross[ei] = cross3(state.vertices[e.a], state.vertices[e.b]);
  }

  const faceVectorArea: Vec3[] = new Array(state.faces.length);
  const faceScalarArea: number[] = new Array(state.faces.length);
  const facePyramidVolume: number[] = new Array(state.faces.length);
  const signedIncidenceArea: number[][] = new Array(state.faces.length);
  const faceCentroid: Vec3[] = new Array(state.faces.length);

  for (let fi = 0; fi < state.faces.length; fi++) {
    const n = planes[fi].n;
    const b = planes[fi].b;
    const incs = topology.edgeIncidencesByFace[fi];
    let a0 = 0;
    let a1 = 0;
    let a2 = 0;
    const incArea: number[] = new Array(incs.length);
    for (let li = 0; li < incs.length; li++) {
      const inc: FaceEdgeIncidence = incs[li];
      const t = edgeCross[inc.edgeIndex];
      const sign = inc.sign;
      a0 += 0.5 * sign * t[0];
      a1 += 0.5 * sign * t[1];
      a2 += 0.5 * sign * t[2];
      incArea[li] = sign * dot3(n, t);
    }
    const a: Vec3 = [a0, a1, a2];
    faceVectorArea[fi] = a;
    const A = dot3(n, a);
    faceScalarArea[fi] = A;
    facePyramidVolume[fi] = (b * A) / 3;
    signedIncidenceArea[fi] = incArea;

    const denom = 6 * A;
    if (Math.abs(denom) <= 1e-12) {
      faceCentroid[fi] = averageVertices(state.vertices, state.faces[fi]);
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

  let centerOfMass: Vec3 = averageVertices(state.vertices);
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
