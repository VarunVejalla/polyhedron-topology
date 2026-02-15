import { useEffect, useMemo, useState } from "react";
import type { NodeId, SimpleGraph } from "../../graph/types";
import { cloneGraph, edgeIdFor } from "../../graph/core";
import { checkPolyhedral } from "../../graph/validity";
import { chooseOuterFace, facesFromEmbedding, planarDualFromFaces, type Face } from "../../graph/embedding";
import { springLayoutStep, tutteLayout } from "../../graph/layout";

type LayoutMode = "manual" | "spring" | "tutte";

type UseGraphEditorStateOpts = {
  graph: SimpleGraph;
  updateGraph: (g: SimpleGraph) => void;
  commitGraph: (g: SimpleGraph) => void;
  canvas: { w: number; h: number; padding: number };
  newNodePrefix: string;
  onSyncDual: (dual: SimpleGraph) => void;
};


// Intentionally avoid a "live ref" pattern here.
// This repo's eslint config enforces react-hooks/refs (no `.current` access during render),
// and GraphEditor handlers are passed through React props, so they naturally get fresh
// closures each render.

function nextNodeId(g: SimpleGraph, prefix: string): NodeId {
  let k = 0;
  while (true) {
    const id = `${prefix}${k}`;
    if (!g.nodes.some((n) => n.id === id)) return id;
    k++;
  }
}

function centroidOfCycle(g: SimpleGraph, cycle: NodeId[]): { x: number; y: number } {
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

export function useGraphEditorState(opts: UseGraphEditorStateOpts) {
  const { graph: g, updateGraph, commitGraph, canvas, newNodePrefix, onSyncDual } = opts;

  // NOTE: We avoid refs-as-state (react-hooks/refs). Instead we recompute the final
  // dragged position on mouse up from the stored pointer coordinates.
  const { w: width, h: height, padding } = canvas;

  const [selectedNodes, setSelectedNodes] = useState<NodeId[]>([]);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  // Dragging is modeled as an *overlay* on top of the committed graph `g`.
  // This keeps heavy derived computations (planarity checks, face extraction, etc.) keyed on `g`
  // and prevents jitter from recomputing them on every pointer move.
  type DragState = { id: NodeId; dx: number; dy: number; curX: number; curY: number };
  const [dragging, setDragging] = useState<DragState | null>(null);
  const [pinned, setPinned] = useState<Set<NodeId>>(() => new Set());


  const [edgeCreateMode, setEdgeCreateMode] = useState(false);
  const [pendingEdgeStart, setPendingEdgeStart] = useState<NodeId | null>(null);

  const [layoutMode, setLayoutMode] = useState<LayoutMode>("manual");
  const [selectedOuterFaceId, setSelectedOuterFaceId] = useState<string | null>(null);

  const topology = useMemo(() => {
    const nodeIds = g.nodes
      .map((n) => n.id)
      .slice()
      .sort();
    // Topology-only edges, but keep stable ids because the planarity/embedding code uses edge ids internally.
    // We canonicalize endpoint order for signature stability; the id is preserved from the committed graph.
    const edgesTopo = g.edges
      .map((e) => {
        const a = e.source < e.target ? e.source : e.target;
        const b = e.source < e.target ? e.target : e.source;
        return { id: e.id, source: a, target: b };
      })
      .slice()
      .sort((e1, e2) => {
        if (e1.source !== e2.source) return e1.source < e2.source ? -1 : 1;
        if (e1.target !== e2.target) return e1.target < e2.target ? -1 : 1;
        return e1.id < e2.id ? -1 : e1.id > e2.id ? 1 : 0;
      });

    const topologySig = `${nodeIds.join("|")}::${edgesTopo.map((e) => `${e.source}--${e.target}`).join("|")}`;

    const topoGraph: SimpleGraph = {
      nodes: nodeIds.map((id) => ({ id, label: id, x: 0, y: 0 })),
      edges: edgesTopo,
    };

    return { nodeIds, edgesTopo, topologySig, topoGraph };
  }, [g.nodes, g.edges]);

  // Validity checks can be relatively expensive; compute them against the committed graph only.
  // Since `g` does not change during drag (only the overlay does), this stays stable while dragging.
  const report = useMemo(() => checkPolyhedral(topology.topoGraph), [topology.topoGraph]);

  const { polyFaces, polyFaceError } = useMemo(() => {
    if (!report.ok) return { polyFaces: null as Face[] | null, polyFaceError: null as string | null };
    try {
      const faces = facesFromEmbedding(topology.nodeIds, topology.edgesTopo, report.embedding);
      return { polyFaces: faces, polyFaceError: null };
    } catch (e: any) {
      return { polyFaces: null, polyFaceError: String(e?.message ?? e) };
    }
  }, [topology.nodeIds, topology.edgesTopo, report]);

  useEffect(() => {
    if (!polyFaces || polyFaces.length === 0) {
      setSelectedOuterFaceId(null);
      return;
    }
    const exists = selectedOuterFaceId ? polyFaces.some((f) => f.id === selectedOuterFaceId) : false;
    if (!exists) {
      const outer = chooseOuterFace(polyFaces);
      setSelectedOuterFaceId(outer?.id ?? polyFaces[0].id);
    }
  }, [polyFaces, selectedOuterFaceId]);

  const setGraph = (next: SimpleGraph, commit: boolean) => {
    updateGraph(next);
    if (commit) commitGraph(next);
  };

  // The graph displayed in the canvas (committed graph + drag overlay).
  const displayGraph: SimpleGraph = useMemo(() => {
    if (!dragging) return g;
    const { id, curX, curY } = dragging;
    // Shallow copy the graph with an updated nodes array.
    return {
      nodes: g.nodes.map((n) => (n.id === id ? { ...n, x: curX, y: curY } : n)),
      edges: g.edges,
    };
  }, [g, dragging]);

  const clearSelection = () => {
    setSelectedNodes([]);
    setSelectedEdgeId(null);
    setPendingEdgeStart(null);
  };

  const onBackgroundDown = () => {
    // Pointer-capture during drags can still deliver background events; ignore them.
    if (dragging) return;
    clearSelection();
  };

  const onNodeDown = (id: NodeId, p: { x: number; y: number }, meta: { shiftKey: boolean; pointerId: number }) => {

    if (edgeCreateMode) {
      if (!pendingEdgeStart) {
        setPendingEdgeStart(id);
        setSelectedNodes([id]);
        return;
      }
      const u = pendingEdgeStart;
      const v = id;
      setPendingEdgeStart(null);
      setSelectedNodes([]);
      // enforce simplicity here: no self-loops, no multi-edges
      if (u === v) return;
      const exists = g.edges.some((ed) => (ed.source === u && ed.target === v) || (ed.source === v && ed.target === u));
      if (exists) return;
      const next = cloneGraph(g);
      next.edges.push({ id: edgeIdFor(u, v), source: u, target: v });
      setGraph(next, true);
      return;
    }

    if (meta.shiftKey) {
      setSelectedNodes((prev: NodeId[]) => {
        const s = new Set(prev);
        if (s.has(id)) s.delete(id);
        else s.add(id);
        return [...s];
      });
      setSelectedEdgeId(null);
      return;
    }

    setSelectedNodes([id]);
    setSelectedEdgeId(null);

    const nd = g.nodes.find((n) => n.id === id);
    if (!nd) return;

    const curX = Math.max(padding, Math.min(width - padding, p.x));
    const curY = Math.max(padding, Math.min(height - padding, p.y));
    setDragging({ id, dx: p.x - nd.x, dy: p.y - nd.y, curX, curY });
  };

  const onPointerMove = (p: { x: number; y: number }) => {
    if (!dragging) return;
    const { dx, dy } = dragging;
    const curX = Math.max(padding, Math.min(width - padding, p.x - dx));
    const curY = Math.max(padding, Math.min(height - padding, p.y - dy));
    // Hot path: only update the overlay state. Do NOT update the committed graph `g`.
    setDragging((prev: DragState | null) => (prev ? { ...prev, curX, curY } : prev));
  };

  const onPointerUp = () => {
    if (!dragging) return;
    const { id, curX, curY } = dragging;
    const next = cloneGraph(g);
    const nd = next.nodes.find((n) => n.id === id);
    if (nd) {
      nd.x = curX;
      nd.y = curY;
    }
    setDragging(null);
    setGraph(next, true);
  };

  const onEdgeClick = (eid: string) => {
    setSelectedNodes([]);
    setSelectedEdgeId(eid);
  };

  const addVertex = () => {
    const id = nextNodeId(g, newNodePrefix);
    const next = cloneGraph(g);
    next.nodes.push({
      id,
      label: id,
      x: padding + Math.random() * (width - 2 * padding),
      y: padding + Math.random() * (height - 2 * padding),
    });
    setGraph(next, true);
  };

  const removeSelected = () => {
    const sel = new Set<NodeId>(selectedNodes);
    if (sel.size === 0 && selectedEdgeId) {
      const next = cloneGraph(g);
      next.edges = next.edges.filter((e) => e.id !== selectedEdgeId);
      setGraph(next, true);
      clearSelection();
      return;
    }

    const next = cloneGraph(g);
    next.nodes = next.nodes.filter((n) => !sel.has(n.id));
    next.edges = next.edges.filter((e) => !sel.has(e.source) && !sel.has(e.target));
    setPinned((prev: Set<NodeId>) => {
      const out = new Set(prev);
      for (const v of sel) out.delete(v);
      return out;
    });
    setGraph(next, true);
    clearSelection();
  };

  const addEdgeBetweenSelected = () => {
    if (selectedNodes.length !== 2) return;
    const [u, v] = selectedNodes;
    if (u === v) return;
    const exists = g.edges.some((e) => (e.source === u && e.target === v) || (e.source === v && e.target === u));
    if (exists) return;
    const next = cloneGraph(g);
    next.edges.push({ id: edgeIdFor(u, v), source: u, target: v });
    setGraph(next, true);
  };

  const pinSelected = (pin: boolean) => {
    setPinned((prev: Set<NodeId>) => {
      const out = new Set(prev);
      for (const v of selectedNodes) {
        if (pin) out.add(v);
        else out.delete(v);
      }
      return out;
    });
  };

  const applyLayout = () => {
    if (layoutMode === "manual") return;

    if (layoutMode === "spring") {
      let next = cloneGraph(g);
      for (let it = 0; it < 40; it++) {
        next = springLayoutStep(next, pinned, canvas, 0.5);
      }
      setGraph(next, true);
      return;
    }

    if (!report.ok) return;
    if (!polyFaces) return;

    const outerFace =
      (selectedOuterFaceId ? polyFaces.find((f) => f.id === selectedOuterFaceId) : null) ??
      chooseOuterFace(polyFaces);
    if (!outerFace) return;

    const pos = tutteLayout(g, outerFace.cycle, canvas, pinned);
    const next = cloneGraph(g);
    for (const n of next.nodes) {
      const p = pos.get(n.id);
      if (p && !pinned.has(n.id)) {
        n.x = p.x;
        n.y = p.y;
      }
    }
    setGraph(next, true);
  };

  const syncDual = () => {
    if (!polyFaces) return;
    const dual = planarDualFromFaces(polyFaces);

    // Place dual vertices at face centroids (current layout).
    const centroids = new Map<string, { x: number; y: number }>();
    for (const f of polyFaces) centroids.set(f.id, centroidOfCycle(g, f.cycle));
    for (const nd of dual.nodes) {
      const c = centroids.get(nd.id);
      if (c) {
        nd.x = c.x;
        nd.y = c.y;
      } else {
        nd.x = padding + Math.random() * (width - 2 * padding);
        nd.y = padding + Math.random() * (height - 2 * padding);
      }
    }

    onSyncDual(dual);
  };

  const toggleEdgeMode = () => {
    clearSelection();
    setEdgeCreateMode((v: boolean) => !v);
  };

  return {
    displayGraph,
    report,
    selectedNodes,
    selectedEdgeId,
    pinned,
    edgeCreateMode,
    pendingEdgeStart,
    layoutMode,
    polyFaces,
    polyFaceError,
    selectedOuterFaceId,
    setSelectedOuterFaceId,
    setLayoutMode,
    onBackgroundDown,
    onPointerMove,
    onPointerUp,
    onNodeDown,
    onEdgeClick,
    addVertex,
    toggleEdgeMode,
    addEdgeBetweenSelected,
    removeSelected,
    pinSelected,
    applyLayout,
    syncDual,
  };
}
