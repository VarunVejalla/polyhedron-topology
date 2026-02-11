import React from "react";
import type { SimpleGraph } from "../../graph/types";
import { GraphCanvas } from "./GraphCanvas";
import { EditPanel } from "./EditPanel";
import { LayoutPanel } from "./LayoutPanel";
import { DualPanel } from "./DualPanel";
import { FacesPanel } from "./FacesPanel";
import { useGraphEditorState } from "./useGraphEditorState";

type Props = {
  title: string;
  graph: SimpleGraph;
  updateGraph: (g: SimpleGraph) => void;
  commitGraph: (g: SimpleGraph) => void;
  onSyncDual: (dual: SimpleGraph) => void;
  canvas: { w: number; h: number; padding: number };
  newNodePrefix: string;
};

export function SingleGraphEditor({ title, graph, updateGraph, commitGraph, onSyncDual, canvas, newNodePrefix }: Props) {
  const st = useGraphEditorState({ graph, updateGraph, commitGraph, canvas, newNodePrefix, onSyncDual });
  const { w: width, h: height } = canvas;

  return (
    <div style={{ border: "1px solid #e6e6e6", borderRadius: 14, padding: 10, background: "#fff" }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8, flexWrap: "wrap" }}>
        <div style={{ fontWeight: 800 }}>{title}</div>
        <span style={{ marginLeft: 6, fontSize: 12, color: "#444" }}>
          {st.report.ok ? <>✅ planar + 3-connected</> : <>⚠ {st.report.reason}</>}
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 260px", gap: 12, alignItems: "start" }}>
        <div>
          <GraphCanvas
            graph={st.displayGraph}
            width={width}
            height={height}
            selectedNodes={st.selectedNodes}
            selectedEdgeId={st.selectedEdgeId}
            pinned={st.pinned}
            edgeCreateMode={st.edgeCreateMode}
            pendingEdgeStart={st.pendingEdgeStart}
            onBackgroundDown={st.onBackgroundDown}
            onPointerMove={st.onPointerMove}
            onPointerUp={st.onPointerUp}
            onNodeDown={st.onNodeDown}
            onEdgeClick={st.onEdgeClick}
          />
        </div>

        <div style={{ display: "grid", gap: 10 }}>
          <EditPanel
            selectedNodes={st.selectedNodes}
            selectedEdgeId={st.selectedEdgeId}
            pinnedCount={st.pinned.size}
            edgeCreateMode={st.edgeCreateMode}
            onAddNode={st.addVertex}
            onToggleEdgeMode={st.toggleEdgeMode}
            onAddEdge={st.addEdgeBetweenSelected}
            onDelete={st.removeSelected}
            onPinSelected={st.pinSelected}
          />

          <LayoutPanel
            isPlanarOk={st.report.ok}
            layoutMode={st.layoutMode}
            setLayoutMode={st.setLayoutMode}
            onApplyLayout={st.applyLayout}
            polyFaceError={st.polyFaceError}
          />

          <DualPanel canSync={!!st.polyFaces} onSyncDual={st.syncDual} />

          <FacesPanel isPlanarOk={st.report.ok} faces={st.polyFaces} polyFaceError={st.polyFaceError} />
        </div>
      </div>
    </div>
  );
}
