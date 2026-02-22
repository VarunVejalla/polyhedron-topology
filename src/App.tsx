import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import "./App.css";

import { PrismEditor, type PrismEditorHandle } from "./components/PrismEditor";
import { GraphEditor } from "./components/GraphEditor";
import { ProjectionSettingsPanel } from "./components/ProjectionSettingsPanel";

import { presetNames } from "./graph/presets";
import { derivePolyFromFaceGraph, derivePolyFromVertexGraph } from "./graph/pipeline";
import { GRAPH_VIEW } from "./graph/view";

import { createInitialState, documentReducer } from "./state/document";
import { toProjectorParams } from "./state/projectionSettings";

export default function App() {
  const presetList = useMemo(() => presetNames(), []);
  const [state, dispatch] = useReducer(documentReducer, undefined, () => createInitialState());
  const doc = state.present;

  const [planarity, setPlanarity] = useState(0);
  const [unitNormality, setUnitNormality] = useState(0);
  const [convexityViolation, setConvexityViolation] = useState(0);
  const [isConvexNow, setIsConvexNow] = useState(true);
  const [handleCount, setHandleCount] = useState(0);
  const [currentVolume, setCurrentVolume] = useState(0);
  const [isComputing, setIsComputing] = useState(false);
  const prismRef = useRef<PrismEditorHandle | null>(null);

  const undo = () => dispatch({ type: "UNDO" });
  const redo = () => dispatch({ type: "REDO" });

  const splitDragRef = useRef<{ dragging: boolean; startX: number; startW: number }>({
    dragging: false,
    startX: 0,
    startW: 0,
  });

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      if (!splitDragRef.current.dragging) return;
      const dx = e.clientX - splitDragRef.current.startX;
      const maxW = Math.max(260, window.innerWidth - 420);
      const w = Math.max(260, Math.min(maxW, splitDragRef.current.startW + dx));
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

  useEffect(() => {
    const clampLeftPane = () => {
      if (!doc.ui.showGraphs || !doc.ui.show3D) return;
      const maxW = Math.max(260, window.innerWidth - 420);
      if (doc.ui.leftWidth > maxW) {
        dispatch({ type: "SET_UI", patch: { leftWidth: maxW } });
      }
    };
    clampLeftPane();
    window.addEventListener("resize", clampLeftPane);
    return () => window.removeEventListener("resize", clampLeftPane);
  }, [doc.ui.showGraphs, doc.ui.show3D, doc.ui.leftWidth]);

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

  const buildPolyFromVertexGraph = () => {
    try {
      const built = derivePolyFromVertexGraph(doc.vertexGraph, GRAPH_VIEW);
      dispatch({
        type: "COMMIT_BUILD",
        patch: {
          vertexGraph: built.vertexGraph,
          faceGraph: built.faceGraph,
          poly: built.poly,
        },
      });
    } catch (e: unknown) {
      alert(String(e));
    }
  };

  const buildPolyFromFaceGraph = () => {
    try {
      const built = derivePolyFromFaceGraph(doc.faceGraph, GRAPH_VIEW);
      dispatch({
        type: "COMMIT_BUILD",
        patch: {
          vertexGraph: built.vertexGraph,
          faceGraph: built.faceGraph,
          poly: built.poly,
        },
      });
    } catch (e: unknown) {
      alert(String(e));
    }
  };

  const toggleViewFlag = (
    key:
      | "showGraphs"
      | "show3D"
      | "showAxes"
      | "showGrid"
      | "showVertexPositions"
      | "showNormals"
      | "showCom"
      | "showProjections"
      | "showStability"
      | "showBasins"
  ) => {
    dispatch({ type: "SET_UI", patch: { [key]: !doc.ui[key] } });
  };

  const runAbortAndRevert = () => {
    prismRef.current?.abortComputation();
    if (state.past.length > 0) dispatch({ type: "UNDO" });
  };

  return (
    <div className="App">
      <div className="toolbar">
        <div className="toolbarSection">
          <label className="toolbarField">
            Preset
            <select value={doc.preset} onChange={(e) => applyPreset(e.target.value)}>
              {presetList.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
          <button className="uiButton uiButtonPrimary" onClick={buildPolyFromVertexGraph} title="Validate vertex graph, sync dual face graph, and build canonical polyhedron">
            Build from vertex
          </button>
          <button className="uiButton uiButtonPrimary" onClick={buildPolyFromFaceGraph} title="Validate face graph, sync dual vertex graph, and build canonical polyhedron">
            Build from face
          </button>
        </div>

        <div className="toolbarSection">
          <button className="uiButton" onClick={undo} disabled={state.past.length === 0} title="Undo (Ctrl/Cmd+Z)">
            Undo
          </button>
          <button className="uiButton" onClick={redo} disabled={state.future.length === 0} title="Redo (Ctrl/Cmd+Y or Ctrl/Cmd+Shift+Z)">
            Redo
          </button>
          <button className="uiButton uiButtonDanger" onClick={runAbortAndRevert} disabled={!isComputing && state.past.length === 0}>
            Abort + Revert
          </button>
        </div>

        <div className="toolbarSection">
          <button className={`uiButton ${doc.ui.showGraphs ? "uiButtonActive" : ""}`} onClick={() => toggleViewFlag("showGraphs")}>
            Graphs
          </button>
          <button className={`uiButton ${doc.ui.show3D ? "uiButtonActive" : ""}`} onClick={() => toggleViewFlag("show3D")}>
            3D
          </button>
        </div>

        <div className="toolbarSection toolbarSectionRight">
          <button className="uiButton" onClick={() => prismRef.current?.hardProject()} disabled={isComputing}>
            Hard project
          </button>
          <div className="statusPill">Planarity: {planarity.toExponential(2)}</div>
          <div className="statusPill">Normality: {unitNormality.toExponential(2)}</div>
          <div className="statusPill">Convexity: {convexityViolation.toExponential(2)}</div>
          <div className="statusPill">Volume: {currentVolume.toExponential(3)}</div>
          <div className={`statusPill ${isConvexNow ? "" : "statusBusy"}`}>{isConvexNow ? "Convex" : "Non-convex"}</div>
          <div className="statusPill">Handles: {handleCount}</div>
          <div className={`statusPill ${isComputing ? "statusBusy" : ""}`}>{isComputing ? "Running" : "Idle"}</div>
          <button
            className="uiButton"
            onClick={() => dispatch({ type: "SET_UI", patch: { showGraphicalSettings: !doc.ui.showGraphicalSettings } })}
            aria-expanded={doc.ui.showGraphicalSettings}
          >
            {doc.ui.showGraphicalSettings ? "Hide graphical" : "Graphical settings"}
          </button>
          <button
            className="uiButton"
            onClick={() => dispatch({ type: "SET_UI", patch: { showAdvancedSettings: !doc.ui.showAdvancedSettings } })}
            aria-expanded={doc.ui.showAdvancedSettings}
          >
            {doc.ui.showAdvancedSettings ? "Hide settings" : "Settings"}
          </button>
        </div>
      </div>

      {doc.ui.showAdvancedSettings && (
        <ProjectionSettingsPanel
          value={doc.projection}
          showAdvanced={doc.ui.showAdvancedProjectionParams}
          onPatch={(patch) => dispatch({ type: "SET_PROJECTION", patch })}
          onShowAdvancedChange={(next) => dispatch({ type: "SET_UI", patch: { showAdvancedProjectionParams: next } })}
        />
      )}

      {doc.ui.showGraphicalSettings && (
        <div className="settingsPanel">
          <div className="settingsDropdown">
            <div className="settingsSectionTitle">Graphical settings</div>
            <div className="settingsButtonGrid">
              <button className={`uiButton ${doc.ui.showVertexPositions ? "uiButtonActive" : ""}`} onClick={() => toggleViewFlag("showVertexPositions")} disabled={isComputing}>
                Vertex positions
              </button>
              <button className={`uiButton ${doc.ui.showAxes ? "uiButtonActive" : ""}`} onClick={() => toggleViewFlag("showAxes")} disabled={isComputing}>
                Axes
              </button>
              <button className={`uiButton ${doc.ui.showGrid ? "uiButtonActive" : ""}`} onClick={() => toggleViewFlag("showGrid")} disabled={isComputing}>
                Grid
              </button>
              <button className={`uiButton ${doc.ui.showNormals ? "uiButtonActive" : ""}`} onClick={() => toggleViewFlag("showNormals")} disabled={isComputing}>
                Normals
              </button>
              <button className={`uiButton ${doc.ui.showCom ? "uiButtonActive" : ""}`} onClick={() => toggleViewFlag("showCom")} disabled={isComputing}>
                COM
              </button>
              <button className={`uiButton ${doc.ui.showProjections ? "uiButtonActive" : ""}`} onClick={() => toggleViewFlag("showProjections")} disabled={isComputing}>
                COM proj
              </button>
              <button className={`uiButton ${doc.ui.showStability ? "uiButtonActive" : ""}`} onClick={() => toggleViewFlag("showStability")} disabled={isComputing}>
                Stability
              </button>
              <button className={`uiButton ${doc.ui.showBasins ? "uiButtonActive" : ""}`} onClick={() => toggleViewFlag("showBasins")} disabled={isComputing}>
                Basins
              </button>
            </div>
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
              ref={prismRef}
              initialVertices={doc.poly.vertices}
              faces={doc.poly.faces}
              method={doc.projection.method}
              params={toProjectorParams(doc.projection)}
              hardProject={{
                mode: doc.projection.hardProjectMode,
                maxIters: doc.projection.hardProjectMaxIters,
                tolPlanar: doc.projection.hardProjectTolPlanar,
              }}
              showAxes={doc.ui.showAxes}
              showGrid={doc.ui.showGrid}
              showNormals={doc.ui.showNormals}
              showCom={doc.ui.showCom}
              showProjections={doc.ui.showProjections}
              showStability={doc.ui.showStability}
              showBasins={doc.ui.showBasins}
              onCommitVertices={(verts) => dispatch({ type: "COMMIT_POLY_VERTICES", vertices: verts })}
              onStatus={(s) => {
                setPlanarity(s.totalPlanarityViolation);
                setHandleCount(s.handleCount);
                setUnitNormality(s.unitNormalityMetric);
                setConvexityViolation(s.convexityViolation);
                setIsConvexNow(s.isConvex);
                setCurrentVolume(s.volume);
              }}
              onRunningChange={setIsComputing}
            />

            <div className="viewportControls">
              <button className="uiButton" onClick={() => prismRef.current?.zoomIn()} disabled={isComputing} title="Zoom in">
                +
              </button>
              <button className="uiButton" onClick={() => prismRef.current?.zoomOut()} disabled={isComputing} title="Zoom out">
                -
              </button>
              <button className="uiButton" onClick={() => prismRef.current?.resetView()} disabled={isComputing}>
                Reset view
              </button>
              <button className={`uiButton ${doc.ui.showVertexPositions ? "uiButtonActive" : ""}`} onClick={() => toggleViewFlag("showVertexPositions")} disabled={isComputing}>
                Vertex positions
              </button>
              <button className="uiButton" onClick={() => prismRef.current?.clearAllHandles()} disabled={isComputing}>
                Clear handles
              </button>
            </div>

            {doc.ui.showVertexPositions && (
              <div className="vertexPanel">
                <div className="vertexPanelTitle">Vertex positions</div>
                <div className="vertexPanelTableWrap">
                  <table className="vertexTable">
                    <thead>
                      <tr>
                        <th>i</th>
                        <th>x</th>
                        <th>y</th>
                        <th>z</th>
                      </tr>
                    </thead>
                    <tbody>
                      {doc.poly.vertices.map((v, i) => (
                        <tr key={i}>
                          <td>{i}</td>
                          <td>{v[0].toFixed(4)}</td>
                          <td>{v[1].toFixed(4)}</td>
                          <td>{v[2].toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
