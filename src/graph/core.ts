import type { GraphEdge, NodeId, SimpleGraph } from "./types";

// ---------- basic graph helpers ----------

export function cloneGraph(g: SimpleGraph): SimpleGraph {
  return {
    nodes: g.nodes.map((n) => ({ ...n })),
    edges: g.edges.map((e) => ({ ...e })),
  };
}

export function buildAdjacency(g: SimpleGraph): Map<NodeId, Set<NodeId>> {
  const adj = new Map<NodeId, Set<NodeId>>();
  for (const n of g.nodes) adj.set(n.id, new Set());
  for (const e of g.edges) {
    if (!adj.has(e.source) || !adj.has(e.target)) continue;
    adj.get(e.source)!.add(e.target);
    adj.get(e.target)!.add(e.source);
  }
  return adj;
}

// ---------- 3-vertex-connectivity (polyhedral prerequisite) ----------

function isConnectedOnAdj(ids: NodeId[], adj: Map<NodeId, Set<NodeId>>, blocked: Set<NodeId>): boolean {
  // Connectivity of the induced subgraph on ids \ blocked.
  let start: NodeId | null = null;
  for (const v of ids) {
    if (!blocked.has(v)) {
      start = v;
      break;
    }
  }
  if (start === null) return true; // empty graph is vacuously connected

  const q: NodeId[] = [start];
  const vis = new Set<NodeId>([start]);
  while (q.length) {
    const u = q.pop()!;
    for (const v of adj.get(u) ?? []) {
      if (blocked.has(v) || vis.has(v)) continue;
      vis.add(v);
      q.push(v);
    }
  }

  // Count unblocked vertices
  let cnt = 0;
  for (const v of ids) if (!blocked.has(v)) cnt++;
  return vis.size === cnt;
}

/**
 * Brute-force 3-vertex-connectivity test (O(n^3)).
 *
 * Returns true iff removing ANY two distinct vertices keeps the graph connected.
 * Caller should enforce n>=4 when using this as a polyhedral gate.
 */
export function isThreeVertexConnected(g: SimpleGraph): boolean {
  const ids = g.nodes.map((n) => n.id);
  const adj = buildAdjacency(g);

  // connectivity prerequisite
  if (!isConnectedOnAdj(ids, adj, new Set())) return false;

  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      const blocked = new Set<NodeId>([ids[i], ids[j]]);
      if (!isConnectedOnAdj(ids, adj, blocked)) return false;
    }
  }
  return true;
}

// ---------- edge utilities ----------

export function edgeIdFor(u: NodeId, v: NodeId): string {
  const a = u < v ? u : v;
  const b = u < v ? v : u;
  return `${a}--${b}`;
}

export function normalizeEdges(edges: Array<[NodeId, NodeId]>): GraphEdge[] {
  const seen = new Set<string>();
  const out: GraphEdge[] = [];
  for (const [u0, v0] of edges) {
    if (u0 === v0) continue;
    const u = u0 < v0 ? u0 : v0;
    const v = u0 < v0 ? v0 : u0;
    const key = `${u}__${v}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ id: `${u}--${v}`, source: u, target: v });
  }
  return out;
}