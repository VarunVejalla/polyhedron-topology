import type { NodeId, SimpleGraph } from "./types";
import { circleLayout } from "./presets";
import { checkPolyhedral } from "./validity";
import { chooseOuterFace, facesFromEmbedding, planarDualFromFaces, type Face } from "./embedding";
import { tutteLayout } from "./layout";
import { buildCanonicalPolyhedron } from "../engine/canonical/canonicalPolyhedron";

export type GraphView = { w: number; h: number; padding: number };

type PolyBuildResult = {
  vertexGraph: SimpleGraph;
  faceGraph: SimpleGraph;
  poly: { vertices: [number, number, number][]; faces: number[][] };
};

function cloneEdges(g: SimpleGraph) {
  return g.edges.map((e) => ({ ...e }));
}

function centroidOfCycle(
  g: { nodes: Array<{ id: NodeId; x: number; y: number }> },
  cycle: NodeId[]
): { x: number; y: number } {
  const byId = new Map<NodeId, { x: number; y: number }>(g.nodes.map((n) => [n.id, { x: n.x, y: n.y }]));
  let sx = 0;
  let sy = 0;
  let cnt = 0;
  for (const v of cycle) {
    const p = byId.get(v);
    if (!p) continue;
    sx += p.x;
    sy += p.y;
    cnt++;
  }
  if (cnt === 0) return { x: 0, y: 0 };
  return { x: sx / cnt, y: sy / cnt };
}

export function deriveDualPairFromVertexGraph(
  g0: SimpleGraph,
  view: GraphView
): { vertexGraph: SimpleGraph; faceGraph: SimpleGraph } | null {
  const rep = checkPolyhedral(g0);
  if (!rep.ok) return null;

  let faces0: Face[];
  try {
    faces0 = facesFromEmbedding(g0.nodes.map((n) => n.id), g0.edges, rep.embedding);
  } catch {
    return null;
  }

  const outerFace = chooseOuterFace(faces0);
  if (!outerFace) return null;

  const posTutte = tutteLayout(g0, outerFace.cycle, view);
  const vertexGraph: SimpleGraph = {
    nodes: g0.nodes.map((n) => {
      const p = posTutte.get(n.id);
      return p ? { ...n, x: p.x, y: p.y } : n;
    }),
    edges: cloneEdges(g0),
  };

  let faceGraph = planarDualFromFaces(faces0);
  const r = Math.max(40, Math.min(view.w, view.h) / 2 - view.padding);
  const posCircle = circleLayout(faceGraph.nodes.map((n) => n.id), view.w, view.h, r);
  faceGraph = {
    nodes: faceGraph.nodes.map((n) => ({ ...n, x: posCircle[n.id].x, y: posCircle[n.id].y })),
    edges: cloneEdges(faceGraph),
  };

  return { vertexGraph, faceGraph };
}

export function derivePolyFromVertexGraph(g0: SimpleGraph, view: GraphView): PolyBuildResult {
  const rep = checkPolyhedral(g0);
  if (!rep.ok) throw new Error("Vertex graph is not a valid polyhedral graph (planar + 3-connected, >= 4 nodes).");

  let faces0: Face[];
  try {
    faces0 = facesFromEmbedding(g0.nodes.map((n) => n.id), g0.edges, rep.embedding);
  } catch (e: unknown) {
    throw new Error(`Face extraction from the planarity embedding failed: ${String(e)}`);
  }

  const outerFace = chooseOuterFace(faces0);
  if (!outerFace) throw new Error("Could not determine an outer face for Tutte layout.");

  const pos = tutteLayout(g0, outerFace.cycle, view);
  const vertexGraph: SimpleGraph = {
    nodes: g0.nodes.map((n) => {
      const p = pos.get(n.id);
      return p ? { ...n, x: p.x, y: p.y } : n;
    }),
    edges: cloneEdges(g0),
  };

  const faceGraph = planarDualFromFaces(faces0);
  const centroids = new Map<string, { x: number; y: number }>();
  for (const f of faces0) centroids.set(f.id, centroidOfCycle(vertexGraph, f.cycle));
  for (const nd of faceGraph.nodes) {
    const c = centroids.get(nd.id);
    if (!c) continue;
    nd.x = c.x;
    nd.y = c.y;
  }

  const realization = buildCanonicalPolyhedron({
    vertexGraph,
    faces: faces0.map((f) => ({ id: f.id, cycle: f.cycle })),
  });

  return {
    vertexGraph,
    faceGraph,
    poly: { vertices: realization.vertices, faces: realization.faces },
  };
}

export function derivePolyFromFaceGraph(g0: SimpleGraph, view: GraphView): PolyBuildResult {
  const rep = checkPolyhedral(g0);
  if (!rep.ok) throw new Error("Face graph is not a valid polyhedral graph (planar + 3-connected, >= 4 nodes).");

  let primalFaces: Face[];
  try {
    primalFaces = facesFromEmbedding(g0.nodes.map((n) => n.id), g0.edges, rep.embedding);
  } catch (e: unknown) {
    throw new Error(`Face extraction from the planarity embedding failed: ${String(e)}`);
  }

  const dual = planarDualFromFaces(primalFaces);
  const repDual = checkPolyhedral(dual);
  if (!repDual.ok) throw new Error("Internal error: computed dual graph did not validate as polyhedral.");

  let dualFaces0: Face[];
  try {
    dualFaces0 = facesFromEmbedding(dual.nodes.map((n) => n.id), dual.edges, repDual.embedding);
  } catch (e: unknown) {
    throw new Error(`Dual face extraction from the planarity embedding failed: ${String(e)}`);
  }

  const outerDualFace = chooseOuterFace(dualFaces0);
  if (!outerDualFace) throw new Error("Could not determine an outer face for Tutte layout of the dual graph.");

  const posDual = tutteLayout(dual, outerDualFace.cycle, view);
  const vertexGraph: SimpleGraph = {
    nodes: dual.nodes.map((n) => {
      const p = posDual.get(n.id);
      return p ? { ...n, x: p.x, y: p.y } : n;
    }),
    edges: cloneEdges(dual),
  };

  const dirEdgeToFace = new Map<string, string>();
  for (const f of primalFaces) {
    const cyc = f.cycle;
    for (let i = 0; i < cyc.length; i++) {
      const a = cyc[i];
      const b = cyc[(i + 1) % cyc.length];
      dirEdgeToFace.set(`${a}__${b}`, f.id);
    }
  }

  const facesForDual: Array<{ id: string; cycle: NodeId[] }> = [];
  for (const nd of g0.nodes) {
    const u = nd.id;
    const rot = rep.embedding[u] ?? [];
    const cyc: NodeId[] = [];
    for (const v of rot) {
      const fid = dirEdgeToFace.get(`${u}__${v}`);
      if (fid) cyc.push(fid);
    }
    if (cyc.length >= 3) facesForDual.push({ id: u, cycle: cyc });
  }
  if (facesForDual.length === 0) throw new Error("Could not derive dual face cycles from the face graph embedding.");

  const realization = buildCanonicalPolyhedron({ vertexGraph, faces: facesForDual });

  return {
    vertexGraph,
    faceGraph: g0,
    poly: { vertices: realization.vertices, faces: realization.faces },
  };
}
