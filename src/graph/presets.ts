import type { NodeId, SimpleGraph } from "./types";
import { normalizeEdges } from "./core";

type VertexGraphPreset = {
  /** Display name shown in the UI. Also used as the key. */
  name: string;
  /** Node ids (labels). */
  nodes: NodeId[];
  /** Undirected adjacency lists (neighbors). */
  adjacency: Record<NodeId, NodeId[]>;
};

function adjFromEdges(nodes: NodeId[], edges: Array<[NodeId, NodeId]>): Record<NodeId, NodeId[]> {
  const m = new Map<NodeId, Set<NodeId>>();
  for (const v of nodes) m.set(v, new Set());
  for (const [a, b] of edges) {
    m.get(a)?.add(b);
    m.get(b)?.add(a);
  }
  const out: Record<NodeId, NodeId[]> = {};
  for (const v of nodes) out[v] = [...(m.get(v) ?? new Set())].sort();
  return out;
}


/**
 * Easy-to-edit preset library.
 *
 * Add a new preset by inserting an entry below. Only the vertex graph is
 * specified; the face graph will be computed automatically as the planar dual
 * once the preset is verified polyhedral.
 */
const VERTEX_GRAPH_PRESETS: Record<string, VertexGraphPreset> = {
  "Triangular prism": (() => {
    const nodes: NodeId[] = ["A", "B", "C", "D", "E", "F"];
    const edges: Array<[NodeId, NodeId]> = [
      // Bottom triangle (B,C,D)
      ["B", "C"],
      ["C", "D"],
      ["D", "B"],
      // Top triangle (A,E,F)
      ["A", "E"],
      ["E", "F"],
      ["F", "A"],
      // Side edges
      ["A", "B"],
      ["C", "E"],
      ["D", "F"],
    ];
    return { name: "Triangular prism", nodes, adjacency: adjFromEdges(nodes, edges) };
  })(),
  "Tetrahedron": (() => {
    const nodes: NodeId[] = ["A", "B", "C", "D"];
    const edges: Array<[NodeId, NodeId]> = [
      ["A", "B"],
      ["A", "C"],
      ["A", "D"],
      ["B", "C"],
      ["B", "D"],
      ["C", "D"],
    ];
    return { name: "Tetrahedron", nodes, adjacency: adjFromEdges(nodes, edges) };
  })(),
  "Cube": (() => {
    const nodes: NodeId[] = ["0", "1", "2", "3", "4", "5", "6", "7"];
    const edges: Array<[NodeId, NodeId]> = [
      // bottom square 0-1-2-3
      ["0", "1"],
      ["1", "2"],
      ["2", "3"],
      ["3", "0"],
      // top square 4-5-6-7
      ["4", "5"],
      ["5", "6"],
      ["6", "7"],
      ["7", "4"],
      // verticals
      ["0", "4"],
      ["1", "5"],
      ["2", "6"],
      ["3", "7"],
    ];
    return { name: "Cube", nodes, adjacency: adjFromEdges(nodes, edges) };
  })(),
  "Octahedron": (() => {
    const nodes: NodeId[] = ["U", "D", "A", "B", "C", "E"];
    const edges: Array<[NodeId, NodeId]> = [
      // U connected to equator
      ["U", "A"],
      ["U", "B"],
      ["U", "C"],
      ["U", "E"],
      // D connected to equator
      ["D", "A"],
      ["D", "B"],
      ["D", "C"],
      ["D", "E"],
      // equator cycle A-B-C-E-A
      ["A", "B"],
      ["B", "C"],
      ["C", "E"],
      ["E", "A"],
    ];
    return { name: "Octahedron", nodes, adjacency: adjFromEdges(nodes, edges) };
  })(),
  "Icosahedron": (() => {
    const nodes: NodeId[] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"];
    const adjacency: Record<NodeId, NodeId[]> = {
      "0": ["2", "4", "6", "8", "10"],
      "1": ["3", "4", "6", "9", "11"],
      "2": ["0", "5", "7", "8", "10"],
      "3": ["1", "5", "7", "9", "11"],
      "4": ["0", "1", "6", "8", "9"],
      "5": ["2", "3", "7", "8", "9"],
      "6": ["0", "1", "4", "10", "11"],
      "7": ["2", "3", "5", "10", "11"],
      "8": ["0", "2", "4", "5", "9"],
      "9": ["1", "3", "4", "5", "8"],
      "10": ["0", "2", "6", "7", "11"],
      "11": ["1", "3", "6", "7", "10"],
    };
    return { name: "Icosahedron", nodes, adjacency };
  })(),
  "Dodecahedron": (() => {
    const nodes: NodeId[] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"];
    const adjacency: Record<NodeId, NodeId[]> = {
      "0": ["8", "12", "16"],
      "1": ["9", "12", "17"],
      "2": ["10", "13", "16"],
      "3": ["11", "13", "17"],
      "4": ["8", "14", "18"],
      "5": ["9", "14", "19"],
      "6": ["10", "15", "18"],
      "7": ["11", "15", "19"],
      "8": ["0", "4", "10"],
      "9": ["1", "5", "11"],
      "10": ["2", "6", "8"],
      "11": ["3", "7", "9"],
      "12": ["0", "1", "14"],
      "13": ["2", "3", "15"],
      "14": ["4", "5", "12"],
      "15": ["6", "7", "13"],
      "16": ["0", "2", "17"],
      "17": ["1", "3", "16"],
      "18": ["4", "6", "19"],
      "19": ["5", "7", "18"],
    };
    return { name: "Dodecahedron", nodes, adjacency };
  })(),
};

export function presetNames(): string[] {
  return Object.keys(VERTEX_GRAPH_PRESETS);
}

export function circleLayout(ids: string[], w: number, h: number, r: number): Record<string, { x: number; y: number }> {
  const out: Record<string, { x: number; y: number }> = {};
  const cx = w / 2;
  const cy = h / 2;
  for (let i = 0; i < ids.length; i++) {
    const t = (2 * Math.PI * i) / ids.length;
    out[ids[i]] = { x: cx + r * Math.cos(t), y: cy + r * Math.sin(t) };
  }
  return out;
}

export function buildVertexPresetGraph(name: string, opts: { w: number; h: number; padding: number }): SimpleGraph {
  const preset = VERTEX_GRAPH_PRESETS[name];
  if (!preset) throw new Error(`Unknown preset: ${name}`);
  const r = Math.max(40, Math.min(opts.w, opts.h) / 2 - opts.padding);
  const pos = circleLayout(preset.nodes, opts.w, opts.h, r);
  const nodes = preset.nodes.map((id) => ({ id, label: id, x: pos[id].x, y: pos[id].y }));

  const edgesRaw: Array<[NodeId, NodeId]> = [];
  for (const u of preset.nodes) {
    for (const v of preset.adjacency[u] ?? []) {
      if (u < v) edgesRaw.push([u, v]);
    }
  }
  const edges = normalizeEdges(edgesRaw);
  return { nodes, edges };
}
