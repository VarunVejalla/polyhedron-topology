import { useCallback, useEffect, useMemo, useRef } from "react";
import type { Vec3 } from "../../engine/math/types";
import { createProjector, type IProjector, type ProjectorParams, type ProjectionMethod } from "../../engine/projection";
import {
  buildPolyFullModel,
  buildPolyState,
  buildPolyTopology,
  type PlaneEq,
  type PolyDerivedCache,
  type PolyRichState,
  type PolyState,
  type PolyTopologyData,
} from "../../engine/poly";
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
  const lastPlanesRef = useRef<PlaneEq[]>([]);
  const topologyRef = useRef<PolyTopologyData>(buildPolyTopology(faces, initialVertices.length));
  const polyStateRef = useRef<PolyState>(buildPolyState(initialVertices, faces));
  const polyRichRef = useRef<PolyRichState>(buildPolyFullModel(polyStateRef.current, topologyRef.current).rich);
  const derivedRef = useRef<PolyDerivedCache>(buildPolyFullModel(polyStateRef.current, topologyRef.current).derived);

  const recomputePolyCache = useCallback((X: ReadonlyArray<Vec3>) => {
    const state = buildPolyState(X, faces, lastPlanesRef.current);
    lastPlanesRef.current = state.facePlanes.map((pl) => ({ n: [pl.n[0], pl.n[1], pl.n[2]], b: pl.b }));
    const full = buildPolyFullModel(state, topologyRef.current);
    polyStateRef.current = full.rich;
    polyRichRef.current = full.rich;
    derivedRef.current = full.derived;
  }, [faces]);

  // Hot-path reads params from a ref.
  useEffect(() => {
    paramsRef.current = params;
    projectorRef.current?.setParams?.({
      rho: params.rho,
      wFree: params.wFree,
      wHandle: params.wHandle,
      lambdaReg: params.lambdaReg,
    });
  }, [params]);

  // Recreate projector atomically from the latest topology/method/vertices.
  useEffect(() => {
    baselineRef.current = initialVertices.map((p: Vec3) => [...p] as Vec3);
    topologyRef.current = buildPolyTopology(faces, baselineRef.current.length);
    handlesRef.current.clear();
    projectorRef.current = createProjector(method, faces, baselineRef.current, paramsRef.current);
    lastPlanesRef.current = [];
    recomputePolyCache(baselineRef.current);
  }, [topologyKey, method, faces, initialVertices, recomputePolyCache]);

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
    recomputePolyCache(proj.getPositionsRef());
  }, [recomputePolyCache]);

  const stepUntilTol = useCallback((maxIters: number, tol: number) => {
    const proj = projectorRef.current;
    if (!proj) return;
    proj.setHandles({ targets: handlesRef.current });
    let it = 0;
    while (it < maxIters) {
      const batch = Math.min(8, maxIters - it);
      proj.step(batch);
      it += batch;
      recomputePolyCache(proj.getPositionsRef());
      if (derivedRef.current.planarityMetric <= tol) break;
    }
    recomputePolyCache(proj.getPositionsRef());
  }, [recomputePolyCache]);

  const getXRef = useCallback((): ReadonlyArray<Vec3> => {
    return projectorRef.current?.getPositionsRef() ?? baselineRef.current;
  }, []);

  const snapshot = useCallback((): Vec3[] => {
    return projectorRef.current?.snapshotPositions() ?? baselineRef.current.map((p) => [...p] as Vec3);
  }, []);

  const commitBaseline = useCallback((snap: Vec3[]) => {
    baselineRef.current = snap;
    projectorRef.current?.reset(snap);
    recomputePolyCache(snap);
  }, [recomputePolyCache]);

  const getPolyState = useCallback((): PolyState => {
    const X = projectorRef.current?.getPositionsRef() ?? baselineRef.current;
    recomputePolyCache(X);
    return polyStateRef.current;
  }, [recomputePolyCache]);

  const getPolyRichState = useCallback((): PolyRichState => {
    const X = projectorRef.current?.getPositionsRef() ?? baselineRef.current;
    recomputePolyCache(X);
    return polyRichRef.current;
  }, [recomputePolyCache]);

  const getDerivedCache = useCallback((): PolyDerivedCache => {
    const X = projectorRef.current?.getPositionsRef() ?? baselineRef.current;
    recomputePolyCache(X);
    return derivedRef.current;
  }, [recomputePolyCache]);

  const diagnostics = useCallback(() => {
    const X = projectorRef.current?.getPositionsRef() ?? baselineRef.current;
    recomputePolyCache(X);
    return { totalPlanarityViolation: derivedRef.current.planarityMetric };
  }, [recomputePolyCache]);

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
      getPolyState,
      getPolyRichState,
      getDerivedCache,
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
      getPolyState,
      getPolyRichState,
      getDerivedCache,
      snapshot,
      commitBaseline,
      diagnostics,
    ]
  );

}
