import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import "./App.css";

import { PrismEditor, type PrismEditorHandle } from "./components/PrismEditor";
import { GraphEditor } from "./components/GraphEditor";

import { projectionMethods, type ProjectionMethod } from "./engine/projection";

import { presetNames } from "./graph/presets";
import { derivePolyFromFaceGraph, derivePolyFromVertexGraph } from "./graph/pipeline";
import { GRAPH_VIEW } from "./graph/view";

import { createInitialState, documentReducer } from "./state/document";

export default function App() {
  const presetList = useMemo(() => presetNames(), []);
  const [state, dispatch] = useReducer(documentReducer, undefined, () => createInitialState());
  const doc = state.present;

  const [planarity, setPlanarity] = useState(0);
  const [unitNormality, setUnitNormality] = useState(0);
  const [convexityViolation, setConvexityViolation] = useState(0);
  const [isConvexNow, setIsConvexNow] = useState(true);
  const [handleCount, setHandleCount] = useState(0);
  const [isComputing, setIsComputing] = useState(false);
  const [threeOnlyHeight, setThreeOnlyHeight] = useState<number | null>(null);
  const prismRef = useRef<PrismEditorHandle | null>(null);
  const mainRowRef = useRef<HTMLDivElement | null>(null);

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
    const updateThreeOnlyHeight = () => {
      const node = mainRowRef.current;
      if (!node) return;
      if (!(doc.ui.show3D && !doc.ui.showGraphs)) {
        setThreeOnlyHeight(null);
        return;
      }
      const top = node.getBoundingClientRect().top;
      const h = Math.max(320, window.innerHeight - top - 10);
      setThreeOnlyHeight(h);
    };
    updateThreeOnlyHeight();
    window.addEventListener("resize", updateThreeOnlyHeight);
    return () => window.removeEventListener("resize", updateThreeOnlyHeight);
  }, [doc.ui.show3D, doc.ui.showGraphs, doc.ui.showAdvancedSettings]);

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
          <div className="statusPill">Planarity: {planarity.toExponential(2)}</div>
          <div className="statusPill">Normality: {unitNormality.toExponential(2)}</div>
          <div className="statusPill">Convexity: {convexityViolation.toExponential(2)}</div>
          <div className={`statusPill ${isConvexNow ? "" : "statusBusy"}`}>{isConvexNow ? "Convex" : "Non-convex"}</div>
          <div className="statusPill">Handles: {handleCount}</div>
          <div className={`statusPill ${isComputing ? "statusBusy" : ""}`}>{isComputing ? "Running" : "Idle"}</div>
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
        <div className="settingsPanel">
          <div className="settingsGrid">
            <label className="toolbarField">
              Method
              <select
                value={doc.projection.method}
                onChange={(e) =>
                  dispatch({
                    type: "SET_PROJECTION",
                    patch: { method: e.target.value as ProjectionMethod },
                  })
                }
              >
                {projectionMethods.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="toolbarField">
              rho
              <input type="number" value={doc.projection.rho} onChange={(e) => dispatch({ type: "SET_PROJECTION", patch: { rho: Number(e.target.value) } })} />
            </label>

            <label className="toolbarField">
              wFree
              <input type="number" value={doc.projection.wFree} onChange={(e) => dispatch({ type: "SET_PROJECTION", patch: { wFree: Number(e.target.value) } })} />
            </label>

            <label className="toolbarField">
              wHandle
              <input type="number" value={doc.projection.wHandle} onChange={(e) => dispatch({ type: "SET_PROJECTION", patch: { wHandle: Number(e.target.value) } })} />
            </label>

            <label className="toolbarField">
              lambdaReg
              <input type="number" value={doc.projection.lambdaReg} onChange={(e) => dispatch({ type: "SET_PROJECTION", patch: { lambdaReg: Number(e.target.value) } })} />
            </label>

            <label className="toolbarField">
              iters/frame
              <input
                type="number"
                value={doc.projection.itersPerFrame}
                onChange={(e) => dispatch({ type: "SET_PROJECTION", patch: { itersPerFrame: Number(e.target.value) } })}
              />
            </label>

            <label className="toolbarField">
              iters/release
              <input
                type="number"
                value={doc.projection.itersOnRelease}
                onChange={(e) => dispatch({ type: "SET_PROJECTION", patch: { itersOnRelease: Number(e.target.value) } })}
              />
            </label>

            <label className="toolbarField">
              hard project mode
              <select
                value={doc.projection.hardProjectMode}
                onChange={(e) =>
                  dispatch({
                    type: "SET_PROJECTION",
                    patch: { hardProjectMode: e.target.value as "iters" | "tol" },
                  })
                }
              >
                <option value="iters">fixed iterations</option>
                <option value="tol">until tolerance</option>
              </select>
            </label>

            <label className="toolbarField">
              hard project max iters
              <input
                type="number"
                min={1}
                step={1}
                value={doc.projection.hardProjectMaxIters}
                onChange={(e) =>
                  dispatch({
                    type: "SET_PROJECTION",
                    patch: { hardProjectMaxIters: Math.max(1, Number(e.target.value)) },
                  })
                }
              />
            </label>

            <label className="toolbarField">
              hard project tolerance
              <input
                type="number"
                min={0}
                step="any"
                value={doc.projection.hardProjectTolPlanar}
                onChange={(e) =>
                  dispatch({
                    type: "SET_PROJECTION",
                    patch: { hardProjectTolPlanar: Math.max(0, Number(e.target.value)) },
                  })
                }
              />
            </label>

            <label className="toolbarField">
              optimize max iters
              <input
                type="number"
                min={1}
                step={1}
                value={doc.projection.optimizeMaxOuterIters}
                onChange={(e) =>
                  dispatch({
                    type: "SET_PROJECTION",
                    patch: { optimizeMaxOuterIters: Math.max(1, Number(e.target.value)) },
                  })
                }
              />
            </label>

            <label className="toolbarField">
              optimize batch
              <input
                type="number"
                min={1}
                step={1}
                value={doc.projection.optimizeBatchIters}
                onChange={(e) =>
                  dispatch({
                    type: "SET_PROJECTION",
                    patch: { optimizeBatchIters: Math.max(1, Number(e.target.value)) },
                  })
                }
              />
            </label>

            <label className="toolbarField">
              optimize rho
              <input
                type="number"
                min={1e-8}
                step="any"
                value={doc.projection.optimizeRho}
                onChange={(e) =>
                  dispatch({
                    type: "SET_PROJECTION",
                    patch: { optimizeRho: Math.max(1e-8, Number(e.target.value)) },
                  })
                }
              />
            </label>

            <label className="toolbarField">
              optimize tol eq
              <input
                type="number"
                min={0}
                step="any"
                value={doc.projection.optimizeTolEq}
                onChange={(e) =>
                  dispatch({
                    type: "SET_PROJECTION",
                    patch: { optimizeTolEq: Math.max(0, Number(e.target.value)) },
                  })
                }
              />
            </label>

            <label className="toolbarField">
              optimize tol ineq
              <input
                type="number"
                min={0}
                step="any"
                value={doc.projection.optimizeTolIneq}
                onChange={(e) =>
                  dispatch({
                    type: "SET_PROJECTION",
                    patch: { optimizeTolIneq: Math.max(0, Number(e.target.value)) },
                  })
                }
              />
            </label>

            <label className="toolbarField">
              optimize stable face (index)
              <input
                type="number"
                min={0}
                max={Math.max(0, doc.poly.faces.length - 1)}
                step={1}
                value={doc.projection.optimizeStableFace}
                onChange={(e) =>
                  dispatch({
                    type: "SET_PROJECTION",
                    patch: {
                      optimizeStableFace: Math.max(0, Math.min(Math.max(0, doc.poly.faces.length - 1), Math.floor(Number(e.target.value)))),
                    },
                  })
                }
              />
            </label>
          </div>
        </div>
      )}

      <div className="mainRow" ref={mainRowRef}>
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
          <div className="rightPane" style={doc.ui.show3D && !doc.ui.showGraphs && threeOnlyHeight ? { height: `${threeOnlyHeight}px` } : undefined}>
            <PrismEditor
              ref={prismRef}
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
              hardProject={{
                mode: doc.projection.hardProjectMode,
                maxIters: doc.projection.hardProjectMaxIters,
                tolPlanar: doc.projection.hardProjectTolPlanar,
              }}
              optimize={{
                maxOuterIters: doc.projection.optimizeMaxOuterIters,
                batchIters: doc.projection.optimizeBatchIters,
                rho: doc.projection.optimizeRho,
                tolEq: doc.projection.optimizeTolEq,
                tolIneq: doc.projection.optimizeTolIneq,
                stableFaceIndex: doc.projection.optimizeStableFace,
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
              <button className="uiButton" onClick={() => prismRef.current?.hardProject()} disabled={isComputing}>
                Hard project
              </button>
              <button className="uiButton" onClick={() => prismRef.current?.optimize()} disabled={isComputing}>
                Optimize
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
              <button
                className={`uiButton ${doc.ui.showProjections ? "uiButtonActive" : ""}`}
                onClick={() => toggleViewFlag("showProjections")}
                disabled={isComputing}
              >
                COM proj
              </button>
              <button className={`uiButton ${doc.ui.showStability ? "uiButtonActive" : ""}`} onClick={() => toggleViewFlag("showStability")} disabled={isComputing}>
                Stability
              </button>
              <button className={`uiButton ${doc.ui.showBasins ? "uiButtonActive" : ""}`} onClick={() => toggleViewFlag("showBasins")} disabled={isComputing}>
                Basins
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
