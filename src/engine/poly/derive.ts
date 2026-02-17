import type { Vec3 } from "../math/types";
import { buildPolyRichState } from "./auxiliary";
import { computePolyLightConstraintMetrics } from "./light";
import type { PolyDerivedCache, PolyRichState, PolyState, RollStep } from "./types";

function dot3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function sub3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function norm3(a: ReadonlyArray<number>): number {
  return Math.hypot(a[0], a[1], a[2]);
}

function edgeKey(a: number, b: number): string {
  return a < b ? `${a},${b}` : `${b},${a}`;
}

function faceCentroid(vertices: ReadonlyArray<Vec3>, face: ReadonlyArray<number>): Vec3 {
  const c: Vec3 = [0, 0, 0];
  if (face.length === 0) return c;
  for (let i = 0; i < face.length; i++) {
    const p = vertices[face[i]];
    c[0] += p[0];
    c[1] += p[1];
    c[2] += p[2];
  }
  const inv = 1 / face.length;
  c[0] *= inv;
  c[1] *= inv;
  c[2] *= inv;
  return c;
}

function polygonSignedMargin(
  vertices: ReadonlyArray<Vec3>,
  face: ReadonlyArray<number>,
  n: ReadonlyArray<number>,
  q: ReadonlyArray<number>
): { margin: number; inside: boolean; minEdgeIndex: number } {
  if (face.length < 3) return { margin: -Infinity, inside: false, minEdgeIndex: -1 };
  const c = faceCentroid(vertices, face);

  let orientSign = 0;
  for (let i = 0; i < face.length; i++) {
    const a = vertices[face[i]];
    const b = vertices[face[(i + 1) % face.length]];
    const e = sub3(b, a);
    const s = dot3(cross3(e, sub3(c, a)), n);
    if (Math.abs(s) > 1e-12) {
      orientSign = s >= 0 ? 1 : -1;
      break;
    }
  }
  if (orientSign === 0) orientSign = 1;

  let minMargin = Number.POSITIVE_INFINITY;
  let minIdx = -1;
  for (let i = 0; i < face.length; i++) {
    const a = vertices[face[i]];
    const b = vertices[face[(i + 1) % face.length]];
    const e = sub3(b, a);
    const len = Math.max(1e-12, norm3(e));
    const s = dot3(cross3(e, sub3(q, a)), n) / len;
    const m = s * orientSign;
    if (m < minMargin) {
      minMargin = m;
      minIdx = i;
    }
  }

  return { margin: minMargin, inside: minMargin >= -1e-9, minEdgeIndex: minIdx };
}

function buildFaceAdjacency(faces: ReadonlyArray<ReadonlyArray<number>>): {
  byEdge: Map<string, number[]>;
  neighbors: Array<Set<number>>;
} {
  const byEdge = new Map<string, number[]>();
  const neighbors: Array<Set<number>> = Array.from({ length: faces.length }, () => new Set<number>());
  for (let fi = 0; fi < faces.length; fi++) {
    const face = faces[fi];
    for (let i = 0; i < face.length; i++) {
      const a = face[i];
      const b = face[(i + 1) % face.length];
      const key = edgeKey(a, b);
      const owners = byEdge.get(key);
      if (!owners) byEdge.set(key, [fi]);
      else owners.push(fi);
    }
  }
  for (const owners of byEdge.values()) {
    if (owners.length !== 2) continue;
    neighbors[owners[0]].add(owners[1]);
    neighbors[owners[1]].add(owners[0]);
  }
  return { byEdge, neighbors };
}

function fiveColor(
  nodes: number[],
  neighbors: Map<number, Set<number>>
): Map<number, number> {
  const color = new Map<number, number>();
  const uncolored = new Set<number>(nodes);

  const chooseNext = (): number => {
    let best = nodes[0];
    let bestSat = -1;
    let bestDeg = -1;
    for (const n of uncolored) {
      const neigh = neighbors.get(n) ?? new Set<number>();
      const used = new Set<number>();
      for (const m of neigh) {
        const c = color.get(m);
        if (c !== undefined) used.add(c);
      }
      const sat = used.size;
      const deg = neigh.size;
      if (sat > bestSat || (sat === bestSat && deg > bestDeg)) {
        best = n;
        bestSat = sat;
        bestDeg = deg;
      }
    }
    return best;
  };

  const canUse = (node: number, c: number): boolean => {
    const neigh = neighbors.get(node);
    if (!neigh) return true;
    for (const m of neigh) {
      if (color.get(m) === c) return false;
    }
    return true;
  };

  const dfs = (): boolean => {
    if (uncolored.size === 0) return true;
    const node = chooseNext();
    uncolored.delete(node);
    for (let c = 0; c < 5; c++) {
      if (!canUse(node, c)) continue;
      color.set(node, c);
      if (dfs()) return true;
      color.delete(node);
    }
    uncolored.add(node);
    return false;
  };

  if (!dfs()) {
    color.clear();
    for (const n of nodes) {
      const used = new Set<number>();
      const neigh = neighbors.get(n) ?? new Set<number>();
      for (const m of neigh) {
        const c = color.get(m);
        if (c !== undefined) used.add(c);
      }
      let c = 0;
      while (used.has(c)) c++;
      color.set(n, c);
    }
  }
  return color;
}

function isRichState(state: PolyState | PolyRichState): state is PolyRichState {
  return "topology" in state && "aux" in state;
}

export function buildPolyDerivedCache(stateArg: PolyState | PolyRichState): PolyDerivedCache {
  const rich = isRichState(stateArg) ? stateArg : buildPolyRichState(stateArg);
  const { vertices, faces, facePlanes, aux } = rich;

  const light = computePolyLightConstraintMetrics(rich);
  const centerOfMass: Vec3 = [aux.centerOfMass[0], aux.centerOfMass[1], aux.centerOfMass[2]];
  const volume = aux.volume;
  const projectedComByFace: Vec3[] = aux.projectedComByFace.map((p) => [p[0], p[1], p[2]] as Vec3);

  const { byEdge, neighbors } = buildFaceAdjacency(faces);

  const stableFace = new Array<boolean>(faces.length).fill(false);
  const signedMarginByFace = new Array<number>(faces.length).fill(0);
  const nextRollByFace: RollStep[] = new Array(faces.length);

  for (let fi = 0; fi < faces.length; fi++) {
    const q = projectedComByFace[fi];
    const face = faces[fi];
    const margin = polygonSignedMargin(vertices, face, facePlanes[fi].n, q);
    stableFace[fi] = margin.inside;
    signedMarginByFace[fi] = margin.margin;

    if (margin.inside || margin.minEdgeIndex < 0) {
      nextRollByFace[fi] = { nextFace: null, edge: null };
      continue;
    }

    const a = face[margin.minEdgeIndex];
    const b = face[(margin.minEdgeIndex + 1) % face.length];
    const owners = byEdge.get(edgeKey(a, b)) ?? [];
    let next: number | null = null;
    if (owners.length >= 2) next = owners[0] === fi ? owners[1] : owners[0];
    nextRollByFace[fi] = { nextFace: next ?? null, edge: [a, b] };
  }

  const settleFaceByFace: Array<number | null> = new Array(faces.length).fill(null);
  const settlePathByFace: number[][] = new Array(faces.length);
  for (let fi = 0; fi < faces.length; fi++) {
    const visited = new Map<number, number>();
    const path: number[] = [fi];
    let cur = fi;
    let settled: number | null = null;
    for (let step = 0; step < faces.length + 2; step++) {
      if (stableFace[cur]) {
        settled = cur;
        break;
      }
      const next = nextRollByFace[cur].nextFace;
      if (next == null) break;
      if (visited.has(next)) {
        path.push(next);
        break;
      }
      visited.set(cur, step);
      path.push(next);
      cur = next;
    }
    settleFaceByFace[fi] = settled;
    settlePathByFace[fi] = path;
  }

  const faceIds = faces.map((_f, fi) => fi);
  const faceAdj = new Map<number, Set<number>>();
  for (let fi = 0; fi < faces.length; fi++) faceAdj.set(fi, neighbors[fi]);
  const defaultColor = fiveColor(faceIds, faceAdj);
  const defaultColorByFace = faceIds.map((fi) => defaultColor.get(fi) ?? 0);

  const basinIdByFace = new Array<number>(faces.length).fill(-1);
  const basinFaces = new Map<number, number[]>();
  for (let fi = 0; fi < faces.length; fi++) {
    const basin = settleFaceByFace[fi] ?? -(fi + 1);
    basinIdByFace[fi] = basin;
    const arr = basinFaces.get(basin);
    if (arr) arr.push(fi);
    else basinFaces.set(basin, [fi]);
  }

  const basinAdj = new Map<number, Set<number>>();
  for (const [bi] of basinFaces) basinAdj.set(bi, new Set<number>());
  for (let fi = 0; fi < faces.length; fi++) {
    const bi = basinIdByFace[fi];
    for (const fj of neighbors[fi]) {
      const bj = basinIdByFace[fj];
      if (bi === bj) continue;
      basinAdj.get(bi)?.add(bj);
      basinAdj.get(bj)?.add(bi);
    }
  }

  const basinColor = fiveColor([...basinFaces.keys()], basinAdj);
  const basinColorByFace = basinIdByFace.map((bid) => basinColor.get(bid) ?? 0);

  return {
    planarityMetric: light.planarityMetric,
    unitNormalityMetric: light.unitNormalityMetric,
    convexityViolation: light.convexityViolation,
    isConvex: light.isConvex,
    centerOfMass,
    volume,
    projectedComByFace,
    stableFace,
    signedMarginByFace,
    nextRollByFace,
    settleFaceByFace,
    settlePathByFace,
    defaultColorByFace,
    basinIdByFace,
    basinColorByFace,
    basinFaces,
  };
}
