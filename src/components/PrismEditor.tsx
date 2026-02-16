import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

import type { Vec3 } from "../engine/math/types";
import type { ProjectorParams, ProjectionMethod } from "../engine/projection";
import { useProjectionController } from "./prismEditor/useProjectionController";
import { useThreePolyhedronScene } from "./prismEditor/useThreePolyhedronScene";
import { usePolyhedronInteraction } from "./prismEditor/usePolyhedronInteraction";

type HardProjectOptions = {
  mode: "iters" | "tol";
  maxIters: number;
  tolPlanar: number;
};

type OptimizeOptions = {
  maxOuterIters: number;
  batchIters: number;
  rho: number;
  tolEq: number;
  tolIneq: number;
  stableFaceIndex: number;
};

type Props = {
  initialVertices: Vec3[];
  faces: number[][];
  method: ProjectionMethod;
  params: ProjectorParams;
  hardProject: HardProjectOptions;
  optimize: OptimizeOptions;
  showAxes: boolean;
  showGrid: boolean;
  showNormals: boolean;
  showCom: boolean;
  showProjections: boolean;
  showStability: boolean;
  showBasins: boolean;
  onCommitVertices?: (verts: Vec3[]) => void;
  onStatus?: (s: {
    totalPlanarityViolation: number;
    handleCount: number;
    unitNormalityMetric: number;
    convexityViolation: number;
    isConvex: boolean;
  }) => void;
  onRunningChange?: (running: boolean) => void;
};

export type PrismEditorHandle = {
  hardProject: () => void;
  optimize: () => void;
  clearAllHandles: () => void;
  abortComputation: () => void;
  revertToBaseline: () => void;
  zoomIn: () => void;
  zoomOut: () => void;
  resetView: () => void;
  setAxesVisible: (visible: boolean) => void;
};

export const PrismEditor = forwardRef<PrismEditorHandle, Props>(function PrismEditor(
  {
    initialVertices,
    faces,
    method,
    params,
    hardProject,
    optimize,
    showAxes,
    showGrid,
    showNormals,
    showCom,
    showProjections,
    showStability,
    showBasins,
    onCommitVertices,
    onStatus,
    onRunningChange,
  }: Props,
  ref
) {
  const mountRef = useRef<HTMLDivElement>(null);

  const controller = useProjectionController(initialVertices, faces, method, params);
  const scene = useThreePolyhedronScene(mountRef, faces, initialVertices);

  const interaction = usePolyhedronInteraction(
    scene,
    controller,
    { showNormals, showCom, showProjections, showStability, showBasins },
    hardProject.mode,
    hardProject.maxIters,
    hardProject.tolPlanar,
    optimize,
    onCommitVertices,
    onStatus,
    onRunningChange
  );

  useEffect(() => {
    scene?.setAxesVisible(showAxes);
  }, [scene, showAxes]);

  useEffect(() => {
    scene?.setGridVisible(showGrid);
  }, [scene, showGrid]);

  useImperativeHandle(
    ref,
    () => ({
      hardProject: () => interaction.hardProject(),
      optimize: () => interaction.optimize(),
      clearAllHandles: () => interaction.clearAllHandles(),
      abortComputation: () => interaction.abortComputation(),
      revertToBaseline: () => interaction.revertToBaseline(),
      zoomIn: () => scene?.zoomBy(0.85),
      zoomOut: () => scene?.zoomBy(1.15),
      resetView: () => scene?.resetView(),
      setAxesVisible: (visible: boolean) => scene?.setAxesVisible(visible),
    }),
    [interaction, scene]
  );

  return (
    <div className="prismEditorRoot">
      <div ref={mountRef} className="prismMount" />
    </div>
  );
});
