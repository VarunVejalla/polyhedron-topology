import { useEffect, useMemo, useRef, useState, useReducer } from "react";
import "./App.css";

import { PrismEditor } from "./components/PrismEditor";
import { GraphEditor } from "./components/GraphEditor";

import { projectionMethods, type ProjectionMethod } from "./engine/projection";
import { buildCanonicalPolyhedron } from "./engine/canonical/canonicalPolyhedron";

import { presetNames } from "./graph/presets";
import type { NodeId } from "./graph/types";
import { checkPolyhedral } from "./graph/validity";
import { facesFromEmbedding, planarDualFromFaces, type Face, chooseOuterFace } from "./graph/embedding";
import { tutteLayout } from "./graph/layout";

import { createInitialState, documentReducer, GRAPH_VIEW } from "./state/document";

export default function App() {
  const presetList = useMemo(() => presetNames(), []);
  const [state, dispatch] = useReducer(documentReducer, undefined, () => createInitialState());
  const doc = state.present;

  // ---- local-only UI state
  const [showHelp, setShowHelp] = useState(false);

  // ---- Status from 3D editor
  const [planarity, setPlanarity] = useState(0);
  const [handleCount, setHandleCount] = useState(0);

  const undo = () => dispatch({ type: "UNDO" });
  const redo = () => dispatch({ type: "REDO" });

  // Split-pane dragging (UI-only; should not affect history)
  const splitDragRef = useRef<{ dragging: boolean; startX: number; startW: number }>({ dragging: false, startX: 0, startW: 0 });

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      if (!splitDragRef.current.dragging) return;
      const dx = e.clientX - splitDragRef.current.startX;
      const w = Math.max(260, splitDragRef.current.startW + dx);
      dispatch({ type: "SET_UI", patch: { leftWidth: w } });
    };
    const onUp = () => {
      splitDragRef.current.dragging = false;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, []);

  // Keyboard shortcuts: undo/redo
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const isMod = e.metaKey || e.ctrlKey;
      if (!isMod) return;
      const k = e.key.toLowerCase();
      if (k === "z" && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if (k === "y" || (k === "z" && e.shiftKey)) {
        e.preventDefault();
        redo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const applyPreset = (preset: string) => dispatch({ type: "APPLY_PRESET", preset });

  const centroidOfCycle = (g: { nodes: Array<{ id: NodeId; x: number; y: number }> }, cycle: NodeId[]): { x: number; y: number } => {
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
  };


  const buildPolyFromVertexGraph = () => {
    const rep = checkPolyhedral(doc.vertexGraph);
    if (!rep.ok) {
      alert("Vertex graph is not a valid polyhedral graph (planar + 3-connected, ≥4 nodes).");
      return;
    }

    // Derive a canonical outer face from the combinatorial planarity embedding,
    // then run Tutte using that facial cycle. This avoids fragile preset/guess outer cycles.
    let faces0: Face[];
    try {
      faces0 = facesFromEmbedding(doc.vertexGraph.nodes.map((n) => n.id), doc.vertexGraph.edges, rep.embedding);
    } catch (e: any) {
      alert(`Face extraction from the planarity embedding failed: ${String(e?.message ?? e)}`);
      return;
    }

    const outerFace = chooseOuterFace(faces0);
    if (!outerFace) {
      alert("Could not determine an outer face for Tutte layout.");
      return;
    }

    const pos = tutteLayout(doc.vertexGraph, outerFace.cycle, GRAPH_VIEW);
    const nodes2 = doc.vertexGraph.nodes.map((n) => {
      const p = pos.get(n.id);
      return p ? { ...n, x: p.x, y: p.y } : n;
    });
    
    const embGraph = { nodes: nodes2, edges: doc.vertexGraph.edges.map((e) => ({ ...e })) };

    // Sync dual (overwrite face graph) from the SAME faces used for building.
    const dual = planarDualFromFaces(faces0);
    const centroids = new Map<string, { x: number; y: number }>();
    for (const f of faces0) centroids.set(f.id, centroidOfCycle(embGraph, f.cycle));
    for (const nd of dual.nodes) {
      const c = centroids.get(nd.id);
      if (c) {
        nd.x = c.x;
        nd.y = c.y;
      }
    }

    const realization = buildCanonicalPolyhedron({
      vertexGraph: embGraph,
      faces: faces0.map((f) => ({ id: f.id, cycle: f.cycle })),
    });

    dispatch({
      type: "COMMIT_BUILD",
      patch: {
        faceGraph: dual,
        poly: { vertices: realization.vertices, faces: realization.faces },
      },
    });
  };

  const buildPolyFromFaceGraph = () => {
    const rep = checkPolyhedral(doc.faceGraph);
    if (!rep.ok) {
      alert("Face graph is not a valid polyhedral graph (planar + 3-connected, ≥4 nodes).");
      return;
    }

    let primalFaces: Face[];
    try {
      primalFaces = facesFromEmbedding(doc.faceGraph.nodes.map((n) => n.id), doc.faceGraph.edges, rep.embedding);
    } catch (e: any) {
      alert(`Face extraction from the planarity embedding failed: ${String(e?.message ?? e)}`);
      return;
    }

    // Build dual = vertex graph.
    const dual = planarDualFromFaces(primalFaces);

    // Give the dual (vertex graph) a canonical planar drawing for stable initialization.
    // We intentionally compute the outer face from the dual's combinatorial embedding.
    const repDual = checkPolyhedral(dual);
    if (!repDual.ok) {
      alert("Internal error: computed dual graph did not validate as polyhedral.");
      return;
    }

    let dualFaces0: Face[];
    try {
      dualFaces0 = facesFromEmbedding(dual.nodes.map((n) => n.id), dual.edges, repDual.embedding);
    } catch (e: any) {
      alert(`Dual face extraction from the planarity embedding failed: ${String(e?.message ?? e)}`);
      return;
    }

    const outerDualFace = chooseOuterFace(dualFaces0);
    if (!outerDualFace) {
      alert("Could not determine an outer face for Tutte layout of the dual graph.");
      return;
    }

    const posDual = tutteLayout(dual, outerDualFace.cycle, GRAPH_VIEW);
    const nodesDual2 = dual.nodes.map((n) => {
      const p = posDual.get(n.id);
      return p ? { ...n, x: p.x, y: p.y } : n;
    });
    const embDualGraph = { nodes: nodesDual2, edges: dual.edges.map((e) => ({ ...e })) };

    // Build face cycles for the dual (vertex graph) by reading, for each primal vertex u,
    // the cyclic list of incident faces to the left of each dart u->v.
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
    for (const nd of doc.faceGraph.nodes) {
      const u = nd.id;
      const rot = rep.embedding[u] ?? [];
      const cyc: NodeId[] = [];
      for (const v of rot) {
        const fid = dirEdgeToFace.get(`${u}__${v}`);
        if (fid) cyc.push(fid);
      }
      // Defensive filter: drop empty/degenerate faces.
      if (cyc.length >= 3) facesForDual.push({ id: u, cycle: cyc });
    }

    if (facesForDual.length === 0) {
      alert("Could not derive dual face cycles from the face graph embedding.");
      return;
    }

    const realization = buildCanonicalPolyhedron({ vertexGraph: embDualGraph, faces: facesForDual });

    dispatch({
      type: "COMMIT_BUILD",
      patch: {
        vertexGraph: embDualGraph,
        poly: { vertices: realization.vertices, faces: realization.faces },
      },
    });
  };

  return (
    <div className="App">
      <div className="toolbar">
        <div className="toolbarGroup">
          <label>
            Preset
            <select value={doc.preset} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 8 }}>
              {presetList.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
          <button
            className="toolbarButton"
            onClick={buildPolyFromFaceGraph}
            style={{ marginLeft: 12 }}
            title="Validate face graph (planar + 3-connected), sync the dual vertex graph, and build the canonical polyhedron"
          >
            Build from face graph
          </button>
          <button
            className="toolbarButton"
            onClick={buildPolyFromVertexGraph}
            title="Validate vertex graph (planar + 3-connected), sync the dual face graph, and build the canonical polyhedron"
          >
            Build from vertex graph
          </button>
        </div>

        <div className="toolbarGroup">
          <button className="toolbarButton" onClick={undo} disabled={state.past.length === 0} title="Undo (Ctrl/Cmd+Z)">
            Undo
          </button>
          <button className="toolbarButton" onClick={redo} disabled={state.future.length === 0} title="Redo (Ctrl/Cmd+Y or Ctrl/Cmd+Shift+Z)">
            Redo
          </button>
        </div>

        <div className="toolbarGroup">
          <label>
            Method
            <select
              value={doc.projection.method}
              onChange={(e) =>
                dispatch({
                  type: "SET_PROJECTION",
                  patch: { method: e.target.value as ProjectionMethod },
                })
              }
              style={{ marginLeft: 8 }}
            >
              {projectionMethods.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="toolbarGroup">
          <label>
            ρ
            <input
              type="number"
              value={doc.projection.rho}
              onChange={(e) => dispatch({ type: "SET_PROJECTION", patch: { rho: Number(e.target.value) } })}
              style={{ width: 80, marginLeft: 8 }}
            />
          </label>
          <label style={{ marginLeft: 12 }}>
            wFree
            <input
              type="number"
              value={doc.projection.wFree}
              onChange={(e) => dispatch({ type: "SET_PROJECTION", patch: { wFree: Number(e.target.value) } })}
              style={{ width: 80, marginLeft: 8 }}
            />
          </label>
          <label style={{ marginLeft: 12 }}>
            wHandle
            <input
              type="number"
              value={doc.projection.wHandle}
              onChange={(e) => dispatch({ type: "SET_PROJECTION", patch: { wHandle: Number(e.target.value) } })}
              style={{ width: 110, marginLeft: 8 }}
            />
          </label>
          <label style={{ marginLeft: 12 }}>
            lambdaReg
            <input
              type="number"
              value={doc.projection.lambdaReg}
              onChange={(e) => dispatch({ type: "SET_PROJECTION", patch: { lambdaReg: Number(e.target.value) } })}
              style={{ width: 100, marginLeft: 8 }}
            />
          </label>
        </div>

        <div className="toolbarGroup">
          <label>
            iters/frame
            <input
              type="number"
              value={doc.projection.itersPerFrame}
              onChange={(e) => dispatch({ type: "SET_PROJECTION", patch: { itersPerFrame: Number(e.target.value) } })}
              style={{ width: 80, marginLeft: 8 }}
            />
          </label>
          <label style={{ marginLeft: 12 }}>
            iters/release
            <input
              type="number"
              value={doc.projection.itersOnRelease}
              onChange={(e) => dispatch({ type: "SET_PROJECTION", patch: { itersOnRelease: Number(e.target.value) } })}
              style={{ width: 90, marginLeft: 8 }}
            />
          </label>
        </div>

        <div className="toolbarGroup" style={{ marginLeft: "auto" }}>
          <button className="toolbarButton" onClick={() => setShowHelp((s) => !s)}>
            {showHelp ? "Hide help" : "Help"}
          </button>
          <label style={{ marginLeft: 10 }}>
            <input type="checkbox" checked={doc.ui.showOverlay} onChange={(e) => dispatch({ type: "SET_UI", patch: { showOverlay: e.target.checked } })} />
            Overlay
          </label>
          <label style={{ marginLeft: 10 }}>
            <input type="checkbox" checked={doc.ui.showGraphs} onChange={(e) => dispatch({ type: "SET_UI", patch: { showGraphs: e.target.checked } })} />
            Graphs
          </label>
          <label style={{ marginLeft: 10 }}>
            <input type="checkbox" checked={doc.ui.show3D} onChange={(e) => dispatch({ type: "SET_UI", patch: { show3D: e.target.checked } })} />
            3D
          </label>
        </div>
      </div>

      {showHelp && (
        <div className="helpPanel">
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
            <div>Planarity (total): {planarity.toExponential(2)}</div>
            <div>Handles: {handleCount}</div>
            <div style={{ opacity: 0.8 }}>Graph edits do not affect the polyhedron until you click “Build polyhedron (canonical)”.</div>
          </div>
        </div>
      )}

      <div className="mainRow">
        {doc.ui.showGraphs && (
          <div className="leftPane" style={{ width: doc.ui.leftWidth }}>
            <GraphEditor
              faceGraph={doc.faceGraph}
              vertexGraph={doc.vertexGraph}
              updateFaceGraph={(g) => dispatch({ type: "LIVE_FACE_GRAPH", graph: g })}
              commitFaceGraph={(g) => dispatch({ type: "COMMIT_FACE_GRAPH", graph: g })}
              updateVertexGraph={(g) => dispatch({ type: "LIVE_VERTEX_GRAPH", graph: g })}
              commitVertexGraph={(g) => dispatch({ type: "COMMIT_VERTEX_GRAPH", graph: g })}
            />
          </div>
        )}

        {doc.ui.showGraphs && doc.ui.show3D && (
          <div
            className="splitter"
            onPointerDown={(e) => {
              splitDragRef.current.dragging = true;
              splitDragRef.current.startX = e.clientX;
              splitDragRef.current.startW = doc.ui.leftWidth;
            }}
            title="Drag to resize"
          />
        )}

        {doc.ui.show3D && (
          <div className="rightPane">
            <PrismEditor
              initialVertices={doc.poly.vertices}
              faces={doc.poly.faces}
              method={doc.projection.method}
              params={{
                rho: doc.projection.rho,
                wFree: doc.projection.wFree,
                wHandle: doc.projection.wHandle,
                lambdaReg: doc.projection.lambdaReg,
                itersPerFrame: doc.projection.itersPerFrame,
                itersOnRelease: doc.projection.itersOnRelease,
              }}
              showOverlay={doc.ui.showOverlay}
              onCommitVertices={(verts) => dispatch({ type: "COMMIT_POLY_VERTICES", vertices: verts })}
              onStatus={(s) => {
                setPlanarity(s.totalPlanarityViolation);
                setHandleCount(s.handleCount);
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
