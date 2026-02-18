import { useCallback, useEffect, useMemo, useRef } from "react";
import type { Vec3 } from "../../engine/math/types";
import { buildPolyDerivedCache, buildPolyState, type PlaneEq, type PolyDerivedCache, type PolyState } from "../../engine/poly";
import { createProjector, type IProjector, type ProjectorParams, type ProjectionMethod } from "../../engine/projection";
import type { ProjectionControllerAPI } from "./types";

function cloneVertices(points: ReadonlyArray<ReadonlyArray<number>>): Vec3[] {
  return points.map((p) => [p[0], p[1], p[2]]);
}

function sameVertices(a: ReadonlyArray<ReadonlyArray<number>>, b: ReadonlyArray<ReadonlyArray<number>>): boolean {
  return a.length === b.length && a.every((p, i) => p[0] === b[i][0] && p[1] === b[i][1] && p[2] === b[i][2]);
}

function sanitizeFaces(faces: ReadonlyArray<ReadonlyArray<number>>, vertexCount: number): number[][] {
  const cleaned: number[][] = [];
  for (let fi = 0; fi < faces.length; fi++) {
    const face = faces[fi].filter((vi) => Number.isInteger(vi) && vi >= 0 && vi < vertexCount);
    if (face.length >= 3) cleaned.push([...face]);
  }
  return cleaned;
}

function clonePlane(plane: PlaneEq): PlaneEq {
  return { n: [plane.n[0], plane.n[1], plane.n[2]], b: plane.b };
}

export function useProjectionController(
  initialVertices: Vec3[],
  faces: number[][],
  method: ProjectionMethod,
  params: ProjectorParams
): ProjectionControllerAPI {
  const cleanFaces = useMemo(() => sanitizeFaces(faces, initialVertices.length), [faces, initialVertices.length]);
  const topologyKey = useMemo(() => JSON.stringify(cleanFaces), [cleanFaces]);

  const projectorRef = useRef<IProjector | null>(null);
  const projectorTopologyKeyRef = useRef(topologyKey);
  const handlesRef = useRef<Map<number, Vec3>>(new Map());
  const baselineRef = useRef<Vec3[]>(cloneVertices(initialVertices));
  const paramsRef = useRef<ProjectorParams>({ ...params });
  const lastPlanesRef = useRef<PlaneEq[]>([]);
  const initialPolyState = useMemo(() => buildPolyState(initialVertices, cleanFaces), [initialVertices, cleanFaces]);
  const polyStateRef = useRef<PolyState>(initialPolyState);
  const derivedRef = useRef<PolyDerivedCache>(buildPolyDerivedCache(initialPolyState));

  const syncDerived = useCallback(
    (points: ReadonlyArray<Vec3>) => {
      const state = buildPolyState(points, cleanFaces, lastPlanesRef.current);
      lastPlanesRef.current = state.facePlanes.map(clonePlane);
      polyStateRef.current = state;
      derivedRef.current = buildPolyDerivedCache(state);
    },
    [cleanFaces]
  );

  useEffect(() => {
    paramsRef.current = { ...params };
    projectorRef.current?.setParams?.({
      rho: params.rho,
      wFree: params.wFree,
      wHandle: params.wHandle,
      itersPerFrame: params.itersPerFrame,
      itersOnRelease: params.itersOnRelease,
    });
  }, [params]);

  useEffect(() => {
    const nextBaseline = cloneVertices(initialVertices);
    if (sameVertices(baselineRef.current, nextBaseline)) return;
    baselineRef.current = nextBaseline;
    handlesRef.current.clear();
    const projector = projectorRef.current;
    const sameTopology = projectorTopologyKeyRef.current === topologyKey;
    const sameVertexCount = (projector?.getPositionsRef().length ?? nextBaseline.length) === nextBaseline.length;
    if (projector && sameTopology && sameVertexCount) projector.reset(nextBaseline);
    syncDerived(nextBaseline);
  }, [initialVertices, topologyKey, syncDerived]);

  useEffect(() => {
    handlesRef.current.clear();
    projectorRef.current = createProjector(method, cleanFaces, baselineRef.current, paramsRef.current);
    projectorTopologyKeyRef.current = topologyKey;
    lastPlanesRef.current = [];
    syncDerived(baselineRef.current);
  }, [method, topologyKey, cleanFaces, syncDerived]);

  const setHandle = useCallback((vid: number, point: Vec3) => {
    handlesRef.current.set(vid, [point[0], point[1], point[2]]);
  }, []);

  const clearHandle = useCallback((vid: number) => {
    handlesRef.current.delete(vid);
  }, []);

  const clearAllHandles = useCallback(() => {
    handlesRef.current.clear();
  }, []);

  const hasHandle = useCallback((vid: number) => handlesRef.current.has(vid), []);

  const getHandleTargets = useCallback((): ReadonlyMap<number, Vec3> => {
    return new Map<number, Vec3>([...handlesRef.current.entries()].map(([k, v]) => [k, [v[0], v[1], v[2]]]));
  }, []);

  const getHandleCount = useCallback(() => handlesRef.current.size, []);

  const getParams = useCallback((): ProjectorParams => ({ ...paramsRef.current }), []);

  const getBaselineSnapshot = useCallback((): Vec3[] => cloneVertices(baselineRef.current), []);

  const resetToBaseline = useCallback(() => {
    const baseline = cloneVertices(baselineRef.current);
    projectorRef.current?.reset(baseline);
    handlesRef.current.clear();
    syncDerived(baseline);
  }, [syncDerived]);

  const step = useCallback(
    (iters: number) => {
      const projector = projectorRef.current;
      if (!projector) return;
      projector.setHandles({ targets: handlesRef.current });
      projector.step(iters);
      syncDerived(projector.getPositionsRef());
    },
    [syncDerived]
  );

  const getXRef = useCallback((): ReadonlyArray<Vec3> => {
    return projectorRef.current?.getPositionsRef() ?? baselineRef.current;
  }, []);

  const getDerivedCache = useCallback((): PolyDerivedCache => derivedRef.current, []);

  const snapshot = useCallback((): Vec3[] => {
    return projectorRef.current?.snapshotPositions() ?? cloneVertices(baselineRef.current);
  }, []);

  const commitBaseline = useCallback(
    (snap: Vec3[]) => {
      const baseline = cloneVertices(snap);
      baselineRef.current = baseline;
      projectorRef.current?.setBaseline(baseline);
      syncDerived(projectorRef.current?.getPositionsRef() ?? baseline);
    },
    [syncDerived]
  );

  const diagnostics = useCallback(() => {
    return { totalPlanarityViolation: derivedRef.current.planarityMetric };
  }, []);

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
      getXRef,
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
      getXRef,
      getDerivedCache,
      snapshot,
      commitBaseline,
      diagnostics,
    ]
  );
}
