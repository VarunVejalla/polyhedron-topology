import { useEffect, useRef } from "react";
import * as THREE from "three";
import type { Vec3 } from "../../engine/math/types";
import type { OverlayOptions, ProjectionControllerAPI, ThreeSceneAPI } from "./types";

export function usePolyhedronInteraction(
  scene: ThreeSceneAPI | null,
  controller: ProjectionControllerAPI,
  overlayOptions: OverlayOptions,
  hardProjectMode: "iters" | "tol",
  hardProjectMaxIters: number,
  hardProjectTolPlanar: number,
  onCommitVertices?: (verts: Vec3[]) => void,
  onStatus?: (s: {
    totalPlanarityViolation: number;
    handleCount: number;
    unitNormalityMetric: number;
    convexityViolation: number;
    isConvex: boolean;
    volume: number;
  }) => void,
  onRunningChange?: (running: boolean) => void
) {
  const stateRef = useRef({
    controller,
    onCommitVertices,
    onStatus,
    onRunningChange,
    overlayOptions,
    hardProjectMode,
    hardProjectMaxIters,
    hardProjectTolPlanar,
  });

  useEffect(() => {
    stateRef.current = {
      controller,
      onCommitVertices,
      onStatus,
      onRunningChange,
      overlayOptions,
      hardProjectMode,
      hardProjectMaxIters,
      hardProjectTolPlanar,
    };
  }, [controller, onCommitVertices, onStatus, onRunningChange, overlayOptions, hardProjectMode, hardProjectMaxIters, hardProjectTolPlanar]);

  useEffect(() => {
    if (!scene) return;
    scene.setDerivedOverlay(controller.getDerivedCache(), overlayOptions);
  }, [scene, controller, overlayOptions]);

  const runningRef = useRef(false);
  const abortRequestedRef = useRef(false);

  const hardProjectRef = useRef<() => void>(() => {});
  const clearAllHandlesRef = useRef<() => void>(() => {});
  const abortComputationRef = useRef<() => void>(() => {});
  const revertToBaselineRef = useRef<() => void>(() => {});

  const setRunning = (running: boolean) => {
    if (runningRef.current === running) return;
    runningRef.current = running;
    stateRef.current.onRunningChange?.(running);
  };

  const pushStatus = (
    diag: { totalPlanarityViolation: number },
    handleCount: number,
    extra: { unitNormalityMetric: number; convexityViolation: number; isConvex: boolean; volume: number }
  ) => {
    stateRef.current.onStatus?.({
      totalPlanarityViolation: diag.totalPlanarityViolation,
      handleCount,
      unitNormalityMetric: extra.unitNormalityMetric,
      convexityViolation: extra.convexityViolation,
      isConvex: extra.isConvex,
      volume: extra.volume,
    });
  };

  useEffect(() => {
    if (!scene) return;

    const {
      renderer,
      camera,
      orbit,
      raycaster,
      mouseNDC,
      vMeshes,
      setMouseFromEvent,
      syncSceneFromX,
      updateSpheresMaterial,
    } = scene;

    let runTimer: number | null = null;
    let disposed = false;

    const cancelRunTimer = () => {
      if (runTimer !== null) {
        window.clearTimeout(runTimer);
        runTimer = null;
      }
    };

    const syncFromController = () => {
      const c = stateRef.current.controller;
      const X = c.getXRef();
      const derived = c.getDerivedCache();
      syncSceneFromX(X);
      scene.setDerivedOverlay(derived, stateRef.current.overlayOptions);
      updateSpheresMaterial(c.getHandleTargets());
      pushStatus({ totalPlanarityViolation: derived.planarityMetric }, c.getHandleCount(), {
        unitNormalityMetric: derived.unitNormalityMetric,
        convexityViolation: derived.convexityViolation,
        isConvex: derived.isConvex,
        volume: derived.volume,
      });
    };

    const applyProjection = (iters: number) => {
      const c = stateRef.current.controller;
      c.step(iters);
      syncFromController();
    };

    const commit = (syncAfterCommit: boolean) => {
      const c = stateRef.current.controller;
      const snap = c.snapshot();
      c.commitBaseline(snap);
      stateRef.current.onCommitVertices?.(snap);
      if (syncAfterCommit) syncFromController();
    };

    const clearAllHandles = () => {
      const c = stateRef.current.controller;
      c.clearAllHandles();
      syncFromController();
    };
    clearAllHandlesRef.current = clearAllHandles;

    const revertToBaseline = () => {
      const c = stateRef.current.controller;
      c.resetToBaseline();
      syncFromController();
    };
    revertToBaselineRef.current = revertToBaseline;

    const abortComputation = () => {
      if (!runningRef.current) return;
      abortRequestedRef.current = true;
    };
    abortComputationRef.current = abortComputation;

    const hardProject = () => {
      if (runningRef.current) return;
      setRunning(true);
      abortRequestedRef.current = false;

      const runController = stateRef.current.controller;
      const maxIters = Math.max(1, Math.floor(stateRef.current.hardProjectMaxIters));
      const tol = Math.max(0, stateRef.current.hardProjectTolPlanar);
      const mode = stateRef.current.hardProjectMode;
      let it = 0;

      const stepBatch = () => {
        if (disposed) return;
        if (stateRef.current.controller !== runController) {
          cancelRunTimer();
          setRunning(false);
          return;
        }

        if (abortRequestedRef.current) {
          abortRequestedRef.current = false;
          cancelRunTimer();
          setRunning(false);
          revertToBaseline();
          return;
        }

        const remaining = maxIters - it;
        if (remaining <= 0) {
          cancelRunTimer();
          commit(false);
          setRunning(false);
          return;
        }

        const batch = Math.min(8, remaining);
        applyProjection(batch);
        it += batch;

        if (mode === "tol" && stateRef.current.controller.getDerivedCache().planarityMetric <= tol) {
          cancelRunTimer();
          commit(false);
          setRunning(false);
          return;
        }

        runTimer = window.setTimeout(stepBatch, 0);
      };

      stepBatch();
    };
    hardProjectRef.current = hardProject;

    const onContextMenu = (e: MouseEvent) => e.preventDefault();
    renderer.domElement.addEventListener("contextmenu", onContextMenu);

    let dragging = false;
    let dragVertex: number | null = null;
    let dragStartClientX = 0;
    let dragStartClientY = 0;
    let dragDidMove = false;
    let dragWasHandle = false;
    let spacePanActive = false;

    const dragPlane = new THREE.Plane();
    const dragHit = new THREE.Vector3();

    const updateLeftMouseMode = (pan: boolean) => {
      orbit.mouseButtons = {
        LEFT: pan ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.PAN,
      };
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code !== "Space") return;
      spacePanActive = true;
      updateLeftMouseMode(true);
      if ((e.target as HTMLElement | null)?.tagName !== "INPUT") e.preventDefault();
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code !== "Space") return;
      spacePanActive = false;
      updateLeftMouseMode(false);
      if ((e.target as HTMLElement | null)?.tagName !== "INPUT") e.preventDefault();
    };

    const onPointerDown = (e: PointerEvent) => {
      if (runningRef.current) return;
      const panOverride = spacePanActive || e.shiftKey;
      updateLeftMouseMode(panOverride);
      if (e.button === 2) {
        e.preventDefault();
        return;
      }
      if (e.button !== 0) return;
      if (panOverride) return;

      setMouseFromEvent(e);
      raycaster.setFromCamera(mouseNDC, camera);

      const hits = raycaster.intersectObjects(vMeshes, false);
      if (hits.length === 0) return;

      const obj = hits[0].object as THREE.Mesh;
      const vid = obj.userData.vertexIndex as number;
      const c = stateRef.current.controller;

      dragging = true;
      dragVertex = vid;
      dragStartClientX = e.clientX;
      dragStartClientY = e.clientY;
      dragDidMove = false;
      dragWasHandle = c.hasHandle(vid);
      orbit.enabled = false;

      const camDir = new THREE.Vector3();
      camera.getWorldDirection(camDir);
      dragPlane.setFromNormalAndCoplanarPoint(camDir, obj.position);

      if (!dragWasHandle) {
        c.setHandle(vid, [obj.position.x, obj.position.y, obj.position.z]);
        syncFromController();
      }
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!dragging || dragVertex === null) return;
      if (!dragDidMove) {
        const dx = e.clientX - dragStartClientX;
        const dy = e.clientY - dragStartClientY;
        if (dx !== 0 || dy !== 0) dragDidMove = true;
      }

      setMouseFromEvent(e);
      raycaster.setFromCamera(mouseNDC, camera);
      if (!raycaster.ray.intersectPlane(dragPlane, dragHit)) return;

      const c = stateRef.current.controller;
      c.setHandle(dragVertex, [dragHit.x, dragHit.y, dragHit.z]);
      applyProjection(c.getParams().itersPerFrame);
    };

    const onPointerUp = () => {
      if (!dragging) return;
      dragging = false;
      const releasedVertex = dragVertex;
      dragVertex = null;
      orbit.enabled = true;

      if (!dragDidMove && releasedVertex !== null) {
        const c = stateRef.current.controller;
        if (dragWasHandle) c.clearHandle(releasedVertex);
        syncFromController();
        return;
      }

      const c = stateRef.current.controller;
      applyProjection(c.getParams().itersOnRelease);
      commit(false);
    };

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);

    syncFromController();

    return () => {
      disposed = true;
      cancelRunTimer();
      setRunning(false);
      renderer.domElement.removeEventListener("contextmenu", onContextMenu);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [scene]);

  return {
    hardProject: () => hardProjectRef.current(),
    clearAllHandles: () => clearAllHandlesRef.current(),
    abortComputation: () => abortComputationRef.current(),
    revertToBaseline: () => revertToBaselineRef.current(),
    isRunning: () => runningRef.current,
  };
}
