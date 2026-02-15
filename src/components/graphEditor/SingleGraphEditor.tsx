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
    <div className="editorCard">
      <div className="editorTitleRow">
        <div className="editorTitle">{title}</div>
        <span className={`editorStatus ${st.report.ok ? "editorStatusOk" : "editorStatusWarn"}`}>
          {st.report.ok ? "OK: planar + 3-connected" : `Issue: ${st.report.reason}`}
        </span>
      </div>

      <div className="editorGrid">
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

        <div className="editorSidePanels">
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

          <details className="editorDetails">
            <summary>Layout</summary>
            <LayoutPanel
              isPlanarOk={st.report.ok}
              layoutMode={st.layoutMode}
              setLayoutMode={st.setLayoutMode}
              onApplyLayout={st.applyLayout}
              polyFaceError={st.polyFaceError}
            />
          </details>

          <details className="editorDetails">
            <summary>Dual sync</summary>
            <DualPanel canSync={!!st.polyFaces} onSyncDual={st.syncDual} />
          </details>

          <details className="editorDetails">
            <summary>Faces</summary>
            <FacesPanel isPlanarOk={st.report.ok} faces={st.polyFaces} polyFaceError={st.polyFaceError} />
          </details>
        </div>
      </div>
    </div>
  );
}
