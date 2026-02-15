import type { SimpleGraph } from "../../graph/types";
import { GraphCanvas } from "./GraphCanvas";
import { EditPanel } from "./EditPanel";
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

          <div className="editorCompactPanel">
            <div className="editorCompactRow">
              <select
                value={st.layoutMode}
                onChange={(e) => st.setLayoutMode(e.target.value as "manual" | "spring" | "tutte")}
                style={{ padding: "6px 8px", borderRadius: 10, border: "1px solid #ddd", flex: 1 }}
              >
                <option value="manual">manual</option>
                <option value="spring">spring</option>
                <option value="tutte" disabled={!st.report.ok}>
                  tutte
                </option>
              </select>
              <button
                onClick={st.applyLayout}
                disabled={st.layoutMode === "manual" || (st.layoutMode === "tutte" && !st.report.ok)}
                className="uiButton"
              >
                Apply
              </button>
            </div>

            {st.layoutMode === "tutte" && st.report.ok && st.polyFaces && (
              <div className="editorCompactRow">
                <select
                  value={st.selectedOuterFaceId ?? ""}
                  onChange={(e) => st.setSelectedOuterFaceId(e.target.value)}
                  style={{ padding: "6px 8px", borderRadius: 10, border: "1px solid #ddd", flex: 1 }}
                >
                  {st.polyFaces.map((f) => (
                    <option key={f.id} value={f.id}>
                      outer face: {f.id}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="editorCompactRow">
              <button onClick={st.syncDual} disabled={!st.polyFaces} className="uiButton" style={{ width: "100%" }}>
                Sync dual
              </button>
            </div>

            {st.layoutMode === "tutte" && st.report.ok && st.polyFaceError && (
              <div style={{ marginTop: 8, fontSize: 12, color: "#c44" }}>{st.polyFaceError}</div>
            )}
          </div>

          <FacesPanel isPlanarOk={st.report.ok} faces={st.polyFaces} polyFaceError={st.polyFaceError} />
        </div>
      </div>
    </div>
  );
}
