import type { Vec3 } from "../math/types";
import type { FaceEdgeIncidence, PlaneEq, PolyAuxState, PolyRichState, PolyState, PolyTopologyData } from "./types";
import { buildPolyTopology } from "./topology";
import { v3 } from "../math/vec3";
import { EPS } from "../math/constants";

function normalizePlane(plane: PlaneEq): PlaneEq {
  const inv = 1 / Math.max(EPS, v3.norm(plane.n));
  return { n: v3.mul(plane.n, inv), b: plane.b * inv };
}

function averageVertices(vertices: ReadonlyArray<Vec3>, ids?: ReadonlyArray<number>): Vec3 {
  let c: Vec3 = [0, 0, 0];
  const n = ids ? ids.length : vertices.length;
  if (n === 0) return c;
  if (ids) {
    for (let i = 0; i < ids.length; i++) {
      c = v3.add(c, vertices[ids[i]]);
    }
  } else {
    for (let i = 0; i < vertices.length; i++) {
      c = v3.add(c, vertices[i]);
    }
  }
  return v3.mul(c, 1 / n);
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
    edgeCross[ei] = v3.cross(state.vertices[e.a], state.vertices[e.b]);
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
    let a: Vec3 = [0, 0, 0];
    const incArea: number[] = new Array(incs.length);
    for (let li = 0; li < incs.length; li++) {
      const inc: FaceEdgeIncidence = incs[li];
      const t = edgeCross[inc.edgeIndex];
      const sign = inc.sign;
      a = v3.add(a, v3.mul(t, 0.5 * sign));
      incArea[li] = sign * v3.dot(n, t);
    }
    faceVectorArea[fi] = a;
    const A = v3.dot(n, a);
    faceScalarArea[fi] = A;
    facePyramidVolume[fi] = (b * A) / 3;
    signedIncidenceArea[fi] = incArea;

    const denom = 6 * A;
    if (Math.abs(denom) <= EPS) {
      faceCentroid[fi] = averageVertices(state.vertices, state.faces[fi]);
    } else {
      let s: Vec3 = [0, 0, 0];
      for (let li = 0; li < incs.length; li++) {
        const inc = incs[li];
        const pI = state.vertices[inc.from];
        const pJ = state.vertices[inc.to];
        const w = incArea[li];
        s = v3.add(s, v3.mul(v3.add(pI, pJ), w));
      }
      faceCentroid[fi] = v3.mul(s, 1 / denom);
    }
  }

  let volume = 0;
  for (let fi = 0; fi < facePyramidVolume.length; fi++) volume += facePyramidVolume[fi];

  let centerOfMass: Vec3 = averageVertices(state.vertices);
  if (Math.abs(volume) > EPS) {
    let s: Vec3 = [0, 0, 0];
    for (let fi = 0; fi < state.faces.length; fi++) {
      const w = facePyramidVolume[fi];
      s = v3.add(s, v3.mul(faceCentroid[fi], w));
    }
    const scale = 3 / (4 * volume);
    centerOfMass = v3.mul(s, scale);
  }

  const projectedComByFace: Vec3[] = new Array(state.faces.length);
  const faceComDistance: number[] = new Array(state.faces.length);
  for (let fi = 0; fi < state.faces.length; fi++) {
    const n = planes[fi].n;
    const b = planes[fi].b;
    const d = v3.dot(n, centerOfMass) - b;
    faceComDistance[fi] = d;
    projectedComByFace[fi] = v3.sub(centerOfMass, v3.mul(n, d));
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
