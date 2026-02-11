import React, { useEffect, useRef, useState } from "react";

import type { Vec3 } from "../engine/math/types";
import type { ProjectorParams, ProjectionMethod } from "../engine/projection";
import { useProjectionController } from "./prismEditor/useProjectionController";
import { useThreePolyhedronScene } from "./prismEditor/useThreePolyhedronScene";
import { usePolyhedronInteraction } from "./prismEditor/usePolyhedronInteraction";

type Props = {
  initialVertices: Vec3[];
  faces: number[][]; // polygon cycles in vertex indices
  method: ProjectionMethod;
  params: ProjectorParams;
  // Whether to show the in-viewport overlay (handle/planarity readout + buttons)
  showOverlay?: boolean;
  // Called after a drag-release commit.
  onCommitVertices?: (verts: Vec3[]) => void;
  // Called whenever diagnostics or handle count updates.
  onStatus?: (s: { totalPlanarityViolation: number; handleCount: number }) => void;
};

export function PrismEditor({ initialVertices, faces, method, params, showOverlay = true, onCommitVertices, onStatus }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);

  const [diagnostic, setDiagnostic] = useState({ totalPlanarityViolation: 0 });
  const [handleCount, setHandleCount] = useState(0);

  const [releaseMode, setReleaseMode] = useState<"iters" | "tol">("iters");
  const [maxReleaseIters, setMaxReleaseIters] = useState(400);
  const [tolPlanar, setTolPlanar] = useState(1e-6);

  const controller = useProjectionController(initialVertices, faces, method, params);
  const scene = useThreePolyhedronScene(mountRef, faces, initialVertices);

  const interaction = usePolyhedronInteraction(
    scene,
    controller,
    releaseMode,
    maxReleaseIters,
    tolPlanar,
    setDiagnostic,
    setHandleCount,
    onCommitVertices,
    onStatus
  );

  // When the scene first becomes ready or the parent supplies new vertices, sync visuals.
  // IMPORTANT: do not sync from baseline here; that would overwrite in-progress drags.
  useEffect(() => {
    if (!scene) return;
    scene.syncSceneFromX(initialVertices);
  }, [scene, initialVertices]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <div ref={mountRef} style={{ position: "absolute", inset: 0 }} />

      {showOverlay && (
        <div
          style={{
            position: "absolute",
            left: 12,
            bottom: 12,
            padding: "10px 12px",
            borderRadius: 12,
            background: "rgba(255,255,255,0.92)",
            border: "1px solid #e6e6e6",
            fontSize: 12,
            color: "#222",
            display: "grid",
            gap: 6,
            maxWidth: 340,
          }}
        >
          <div>
            <b>Polyhedron editor</b>
          </div>
          <div>Drag a vertex (LMB) to move it. Right-click a vertex to clear its handle.</div>
          <div>
            Handles: <b>{handleCount}</b> · Total planarity violation:{" "}
            <b>{diagnostic.totalPlanarityViolation.toExponential(2)}</b>
          </div>
          <div style={{ display: "grid", gap: 6 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <span>Release:</span>
              <select value={releaseMode} onChange={(e) => setReleaseMode(e.target.value as "iters" | "tol")}>
                <option value="iters">fixed iterations</option>
                <option value="tol">until tol</option>
              </select>
              <label style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
                max iters
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={maxReleaseIters}
                  onChange={(e) => setMaxReleaseIters(Math.max(1, Number(e.target.value)))}
                  style={{ width: 90 }}
                />
              </label>
              <label style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
                tol
                <input
                  type="number"
                  min={0}
                  step="any"
                  value={tolPlanar}
                  onChange={(e) => setTolPlanar(Math.max(0, Number(e.target.value)))}
                  style={{ width: 90 }}
                />
              </label>
            </div>
            <button
              style={{ padding: "6px 8px", borderRadius: 10, border: "1px solid #ddd", background: "white", cursor: "pointer" }}
              onClick={() => interaction.hardProject()}
              title="Run a convergence pass and commit the result as the new baseline"
            >
              Hard project
            </button>
          </div>
          <button
            style={{ padding: "6px 8px", borderRadius: 10, border: "1px solid #ddd", background: "white", cursor: "pointer" }}
            onClick={() => interaction.clearAllHandles()}
          >
            Clear all handles
          </button>
        </div>
      )}
    </div>
  );
}
