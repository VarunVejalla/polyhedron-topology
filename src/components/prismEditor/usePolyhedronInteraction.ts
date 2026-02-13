import { useEffect, useRef } from "react";
import * as THREE from "three";
import type { Vec3 } from "../../engine/math/types";
import type { ProjectionControllerAPI, ThreeSceneAPI } from "./types";

/**
 * Owns all pointer + keyboard interaction. Critical property:
 * event listeners must remain stable during drag (no bind/unbind on React rerenders).
 */
export function usePolyhedronInteraction(
  scene: ThreeSceneAPI | null,
  controller: ProjectionControllerAPI,
  hardProjectMode: "iters" | "tol",
  hardProjectMaxIters: number,
  hardProjectTolPlanar: number,
  setDiagnostic: (d: { totalPlanarityViolation: number }) => void,
  setHandleCount: (n: number) => void,
  onCommitVertices?: (verts: Vec3[]) => void,
  onStatus?: (s: { totalPlanarityViolation: number; handleCount: number }) => void
) {
  // Capture the latest values in a stable ref for event handlers
  const stateRef = useRef({
    controller,
    setDiagnostic,
    setHandleCount,
    onCommitVertices,
    onStatus,
    hardProjectMode,
    hardProjectMaxIters,
    hardProjectTolPlanar,
  });
  
  // Update ref during render (this is safe and allowed by React)
  stateRef.current = {
    controller,
    setDiagnostic,
    setHandleCount,
    onCommitVertices,
    onStatus,
    hardProjectMode,
    hardProjectMaxIters,
    hardProjectTolPlanar,
  };

  const pushStatus = (diag: { totalPlanarityViolation: number }, handleCount: number) => {
    stateRef.current.setDiagnostic(diag);
    stateRef.current.onStatus?.({ totalPlanarityViolation: diag.totalPlanarityViolation, handleCount });
  };

  const hardProjectRef = useRef<() => void>(() => {});
  const clearAllHandlesRef = useRef<() => void>(() => {});

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

    const onContextMenu = (e: MouseEvent) => e.preventDefault();
    renderer.domElement.addEventListener("contextmenu", onContextMenu);

    // Drag state
    let dragging = false;
    let dragVertex: number | null = null;
    let dragStartClientX = 0;
    let dragStartClientY = 0;
    let dragDidMove = false;
    let dragWasHandle = false;

    const dragPlane = new THREE.Plane();
    const dragHit = new THREE.Vector3();

    const applyProjection = (iters: number) => {
      const c = stateRef.current.controller;
      c.step(iters);
      const X = c.getXRef();
      syncSceneFromX(X);
      const hc = c.getHandleCount();
      stateRef.current.setHandleCount(hc);
      pushStatus(c.diagnostics(), hc);
    };

    const applyProjectionUntilTol = (maxIters: number, tol: number) => {
      const c = stateRef.current.controller;
      c.stepUntilTol(maxIters, tol);
      const X = c.getXRef();
      syncSceneFromX(X);
      const hc = c.getHandleCount();
      stateRef.current.setHandleCount(hc);
      pushStatus(c.diagnostics(), hc);
    };

    const commit = () => {
      const c = stateRef.current.controller;
      const snap = c.snapshot();
      c.commitBaseline(snap);
      stateRef.current.onCommitVertices?.(snap);

      const hc = c.getHandleCount();
      stateRef.current.setHandleCount(hc);
      pushStatus(c.diagnostics(), hc);
    };

    const clearAllHandles = () => {
      const c = stateRef.current.controller;
      c.clearAllHandles();
      updateSpheresMaterial(c.handlesRef.current);
      const hc = c.getHandleCount();
      stateRef.current.setHandleCount(hc);
      pushStatus(c.diagnostics(), hc);
    };
    clearAllHandlesRef.current = clearAllHandles;

    const hardProject = () => {
      const { hardProjectMode, hardProjectMaxIters, hardProjectTolPlanar } = stateRef.current;
      if (hardProjectMode === "tol") applyProjectionUntilTol(hardProjectMaxIters, hardProjectTolPlanar);
      else applyProjection(hardProjectMaxIters);
      commit();
    };
    hardProjectRef.current = hardProject;

    const onPointerDown = (e: PointerEvent) => {
      if (e.button === 2) e.preventDefault();

      setMouseFromEvent(e);
      raycaster.setFromCamera(mouseNDC, camera);

      const hits = raycaster.intersectObjects(vMeshes, false);
      if (hits.length === 0) {
        return;
      }

      const obj = hits[0].object as THREE.Mesh;
      const vid = obj.userData.vertexIndex as number;

      const c = stateRef.current.controller;

      // Right click clears handle
      if (e.button === 2) {
        c.clearHandle(vid);
        updateSpheresMaterial(c.handlesRef.current);
        applyProjection(c.paramsRef.current.itersPerFrame);
        return;
      }

      dragging = true;
      dragVertex = vid;
      dragStartClientX = e.clientX;
      dragStartClientY = e.clientY;
      dragDidMove = false;
      dragWasHandle = c.handlesRef.current.has(vid);
      orbit.enabled = false;

      const camDir = new THREE.Vector3();
      camera.getWorldDirection(camDir);

      dragPlane.setFromNormalAndCoplanarPoint(camDir, obj.position);

      if (!dragWasHandle) {
        c.setHandle(vid, [obj.position.x, obj.position.y, obj.position.z]);
        updateSpheresMaterial(c.handlesRef.current);
        stateRef.current.setHandleCount(c.getHandleCount());
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

      updateSpheresMaterial(c.handlesRef.current);
      applyProjection(c.paramsRef.current.itersPerFrame);
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
        updateSpheresMaterial(c.handlesRef.current);
        const hc = c.getHandleCount();
        stateRef.current.setHandleCount(hc);
        pushStatus(c.diagnostics(), hc);
        return;
      }

      const c = stateRef.current.controller;
      applyProjection(c.paramsRef.current.itersOnRelease);

      commit();
    };

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);

    return () => {
      renderer.domElement.removeEventListener("contextmenu", onContextMenu);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
    };
  }, [scene]);

  return {
    hardProject: () => hardProjectRef.current(),
    clearAllHandles: () => clearAllHandlesRef.current(),
  };
}
