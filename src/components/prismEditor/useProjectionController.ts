import { useCallback, useEffect, useMemo, useRef } from "react";
import type { Vec3 } from "../../engine/math/types";
import { createProjector, type IProjector, type ProjectorParams, type ProjectionMethod } from "../../engine/projection";
import type { ProjectionControllerAPI } from "./types";

export function useProjectionController(
  initialVertices: Vec3[],
  faces: number[][],
  method: ProjectionMethod,
  params: ProjectorParams
): ProjectionControllerAPI {
  const topologyKey = useMemo(() => JSON.stringify(faces), [faces]);

  const projectorRef = useRef<IProjector | null>(null);
  const handlesRef = useRef<Map<number, Vec3>>(new Map());
  const baselineRef = useRef<Vec3[]>(initialVertices.map((p: Vec3) => [...p] as Vec3));
  const paramsRef = useRef<ProjectorParams>(params);

  // Hot-path reads params from a ref.
  useEffect(() => {
    paramsRef.current = params;
    projectorRef.current?.setParams?.({ rho: params.rho, wFree: params.wFree, wHandle: params.wHandle });
  }, [params]);

  // If parent supplies a new set of vertices (undo/redo, commit), update baseline + reset projector.
  useEffect(() => {
    baselineRef.current = initialVertices.map((p: Vec3) => [...p] as Vec3);
    projectorRef.current?.reset(baselineRef.current);
  }, [initialVertices]);

  // Recreate projector when topology or method changes.
  // Use the latest initialVertices directly instead of from a ref.
  useEffect(() => {
    baselineRef.current = initialVertices.map((p: Vec3) => [...p] as Vec3);
    handlesRef.current.clear();
    projectorRef.current = createProjector(method, faces, baselineRef.current, paramsRef.current);
  }, [topologyKey, method, faces, initialVertices]);

  const setHandle = useCallback((vid: number, p: Vec3) => {
    handlesRef.current.set(vid, [...p] as Vec3);
  }, []);

  const clearHandle = useCallback((vid: number) => {
    handlesRef.current.delete(vid);
  }, []);

  const clearAllHandles = useCallback(() => {
    handlesRef.current.clear();
  }, []);

  const getHandleCount = useCallback(() => handlesRef.current.size, []);

  const step = useCallback((iters: number) => {
    const proj = projectorRef.current;
    if (!proj) return;
    proj.setHandles({ targets: handlesRef.current });
    proj.step(iters);
  }, []);

  const stepUntilTol = useCallback((maxIters: number, tol: number) => {
    const proj = projectorRef.current;
    if (!proj) return;
    proj.setHandles({ targets: handlesRef.current });
    let it = 0;
    while (it < maxIters) {
      const batch = Math.min(8, maxIters - it);
      proj.step(batch);
      it += batch;
      const d = proj.diagnostics();
      if (d.totalPlanarityViolation <= tol) break;
    }
  }, []);

  const getXRef = useCallback((): ReadonlyArray<Vec3> => {
    return projectorRef.current?.getPositionsRef() ?? baselineRef.current;
  }, []);

  const snapshot = useCallback((): Vec3[] => {
    return projectorRef.current?.snapshotPositions() ?? baselineRef.current.map((p) => [...p] as Vec3);
  }, []);

  const commitBaseline = useCallback((snap: Vec3[]) => {
    baselineRef.current = snap;
    projectorRef.current?.reset(snap);
  }, []);

  const diagnostics = useCallback(() => {
    return projectorRef.current?.diagnostics() ?? { totalPlanarityViolation: 0 };
  }, []);

  // Keep the returned controller API object stable across renders.
  return useMemo(
    () => ({
      projectorRef,
      paramsRef,
      handlesRef,
      baselineRef,
      setHandle,
      clearHandle,
      clearAllHandles,
      getHandleCount,
      step,
      stepUntilTol,
      getXRef,
      snapshot,
      commitBaseline,
      diagnostics,
    }),
    [
      setHandle,
      clearHandle,
      clearAllHandles,
      getHandleCount,
      step,
      stepUntilTol,
      getXRef,
      snapshot,
      commitBaseline,
      diagnostics,
    ]
  );

}
