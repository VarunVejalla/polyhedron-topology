import type { FaceEdgeIncidence, PolyEdge, PolyTopologyData, VertexFaceIncidence } from "./types";

function edgeKey(a: number, b: number): string {
  return a < b ? `${a},${b}` : `${b},${a}`;
}

function canonicalEdge(a: number, b: number): [number, number] {
  return a < b ? [a, b] : [b, a];
}

export function buildPolyTopology(
  faces: ReadonlyArray<ReadonlyArray<number>>,
  vertexCount: number
): PolyTopologyData {
  const edgeIndexByKey = new Map<string, number>();
  const edges: PolyEdge[] = [];

  const edgeIncidencesByFace: FaceEdgeIncidence[][] = new Array(faces.length);
  const edgeIncidencesFlat: FaceEdgeIncidence[] = [];

  for (let fi = 0; fi < faces.length; fi++) {
    const face = faces[fi];
    const out: FaceEdgeIncidence[] = [];
    for (let li = 0; li < face.length; li++) {
      const from = face[li];
      const to = face[(li + 1) % face.length];
      const [a, b] = canonicalEdge(from, to);
      const key = edgeKey(from, to);
      let edgeIndex = edgeIndexByKey.get(key);
      if (edgeIndex === undefined) {
        edgeIndex = edges.length;
        edgeIndexByKey.set(key, edgeIndex);
        edges.push({ edgeIndex, a, b });
      }
      const sign: -1 | 1 = from === a && to === b ? 1 : -1;
      const inc: FaceEdgeIncidence = {
        fi,
        localEdge: li,
        from,
        to,
        edgeIndex,
        edgeA: a,
        edgeB: b,
        sign,
      };
      out.push(inc);
      edgeIncidencesFlat.push(inc);
    }
    edgeIncidencesByFace[fi] = out;
  }

  const incidencePairs: VertexFaceIncidence[] = [];
  const nonIncidencePairs: VertexFaceIncidence[] = [];

  for (let fi = 0; fi < faces.length; fi++) {
    const faceSet = new Set<number>(faces[fi]);
    for (const vi of faceSet) incidencePairs.push({ fi, vi });
    for (let vi = 0; vi < vertexCount; vi++) {
      if (!faceSet.has(vi)) nonIncidencePairs.push({ fi, vi });
    }
  }

  return {
    vertexCount,
    faces: faces.map((f) => [...f]),
    edges,
    edgeIncidencesByFace,
    edgeIncidencesFlat,
    incidencePairs,
    nonIncidencePairs,
  };
}

