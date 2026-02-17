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

const cloneVertices = (points: ReadonlyArray<Vec3>): Vec3[] => points.map((p) => [p[0], p[1], p[2]] as Vec3);
const sameVertices = (a: ReadonlyArray<Vec3>, b: ReadonlyArray<Vec3>): boolean =>
  a.length === b.length && a.every((p, i) => p[0] === b[i][0] && p[1] === b[i][1] && p[2] === b[i][2]);

export function useProjectionController(
  initialVertices: Vec3[],
  faces: number[][],
  method: ProjectionMethod,
  params: ProjectorParams
): ProjectionControllerAPI {
  const topologyKey = useMemo(() => JSON.stringify(faces), [faces]);
  const facesForTopology = useMemo(() => JSON.parse(topologyKey) as number[][], [topologyKey]);
  const initialModel = useMemo(() => {
    const topology = buildPolyTopology(facesForTopology, initialVertices.length);
    const full = buildPolyFullModel(buildPolyState(initialVertices, facesForTopology), topology);
    return { topology, full };
  }, [initialVertices, facesForTopology]);
  const projectorRef = useRef<IProjector | null>(null);
  const projectorTopologyKeyRef = useRef(topologyKey);
  const handlesRef = useRef<Map<number, Vec3>>(new Map());
  const baselineRef = useRef<Vec3[]>(cloneVertices(initialVertices));
  const paramsRef = useRef<ProjectorParams>(params);
  const lastPlanesRef = useRef<PlaneEq[]>([]);
  const topologyRef = useRef<PolyTopologyData>(initialModel.topology);
  const polyStateRef = useRef<PolyState>(initialModel.full.rich);
  const polyRichRef = useRef<PolyRichState>(initialModel.full.rich);
  const derivedRef = useRef<PolyDerivedCache>(initialModel.full.derived);

  const recomputePolyCache = useCallback((X: ReadonlyArray<Vec3>) => {
    const state = buildPolyState(X, facesForTopology, lastPlanesRef.current);
    lastPlanesRef.current = state.facePlanes.map((pl) => ({ n: [pl.n[0], pl.n[1], pl.n[2]], b: pl.b }));
    const full = buildPolyFullModel(state, topologyRef.current);
    polyStateRef.current = full.rich;
    polyRichRef.current = full.rich;
    derivedRef.current = full.derived;
  }, [facesForTopology]);

  // Hot-path reads params from a ref.
  useEffect(() => {
    paramsRef.current = {
      rho: params.rho,
      wFree: params.wFree,
      wHandle: params.wHandle,
      lambdaReg: params.lambdaReg,
      itersPerFrame: params.itersPerFrame,
      itersOnRelease: params.itersOnRelease,
    };
    projectorRef.current?.setParams?.({
      rho: params.rho,
      wFree: params.wFree,
      wHandle: params.wHandle,
      lambdaReg: params.lambdaReg,
    });
  }, [params.rho, params.wFree, params.wHandle, params.lambdaReg, params.itersPerFrame, params.itersOnRelease]);

  // Sync external baseline vertices without rebuilding topology/projector.
  useEffect(() => {
    const nextBaseline = cloneVertices(initialVertices);
    if (sameVertices(baselineRef.current, nextBaseline)) return;
    baselineRef.current = nextBaseline;
    handlesRef.current.clear();
    if (projectorTopologyKeyRef.current !== topologyKey) return;
    projectorRef.current?.reset(nextBaseline);
    recomputePolyCache(nextBaseline);
  }, [initialVertices, topologyKey, recomputePolyCache]);

  // Recreate projector atomically only when topology/method changes.
  useEffect(() => {
    topologyRef.current = buildPolyTopology(facesForTopology, baselineRef.current.length);
    handlesRef.current.clear();
    projectorRef.current = createProjector(method, facesForTopology, baselineRef.current, paramsRef.current);
    projectorTopologyKeyRef.current = topologyKey;
    lastPlanesRef.current = [];
    recomputePolyCache(baselineRef.current);
  }, [topologyKey, method, facesForTopology, recomputePolyCache]);

  const setHandle = useCallback((vid: number, p: Vec3) => {
    handlesRef.current.set(vid, [...p] as Vec3);
  }, []);

  const clearHandle = useCallback((vid: number) => {
    handlesRef.current.delete(vid);
  }, []);

  const clearAllHandles = useCallback(() => {
    handlesRef.current.clear();
  }, []);

  const hasHandle = useCallback((vid: number) => handlesRef.current.has(vid), []);

  const getHandleTargets = useCallback((): ReadonlyMap<number, Vec3> => handlesRef.current, []);

  const getHandleCount = useCallback(() => handlesRef.current.size, []);

  const getParams = useCallback((): ProjectorParams => paramsRef.current, []);

  const getBaselineSnapshot = useCallback((): Vec3[] => baselineRef.current.map((p) => [...p] as Vec3), []);

  const resetToBaseline = useCallback(() => {
    const baseline = baselineRef.current.map((p) => [...p] as Vec3);
    projectorRef.current?.reset(baseline);
    handlesRef.current.clear();
    recomputePolyCache(baseline);
  }, [recomputePolyCache]);

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
    const baseline = snap.map((p) => [...p] as Vec3);
    baselineRef.current = baseline;
    projectorRef.current?.reset(baseline);
    recomputePolyCache(baseline);
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
      setHandle,
      clearHandle,
      clearAllHandles,
      hasHandle,
      getHandleTargets,
      getHandleCount,
      getParams,
      getBaselineSnapshot,
      resetToBaseline,
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
      hasHandle,
      getHandleTargets,
      getHandleCount,
      getParams,
      getBaselineSnapshot,
      resetToBaseline,
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
