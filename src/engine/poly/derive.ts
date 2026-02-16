import type { Vec3 } from "../math/types";
import type { PlaneEq, PolyDerivedCache, PolyState, RollStep } from "./types";

function dot3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function sub3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function add3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function mul3(a: ReadonlyArray<number>, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
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

function projectPointToPlane(p: ReadonlyArray<number>, plane: PlaneEq): Vec3 {
  const d = dot3(plane.n, p) - plane.b;
  return [p[0] - plane.n[0] * d, p[1] - plane.n[1] * d, p[2] - plane.n[2] * d];
}

function computeVolumeAndCom(state: PolyState): { volume: number; centerOfMass: Vec3 } {
  const { vertices, faces, facePlanes } = state;
  const ref: Vec3 = [0, 0, 0];
  for (let i = 0; i < vertices.length; i++) {
    ref[0] += vertices[i][0];
    ref[1] += vertices[i][1];
    ref[2] += vertices[i][2];
  }
  if (vertices.length > 0) {
    const inv = 1 / vertices.length;
    ref[0] *= inv;
    ref[1] *= inv;
    ref[2] *= inv;
  }

  let totalVol = 0;
  let comNum: Vec3 = [0, 0, 0];
  for (let fi = 0; fi < faces.length; fi++) {
    const face = faces[fi];
    if (face.length < 3) continue;
    const n = facePlanes[fi].n;
    const a0 = vertices[face[0]];
    for (let i = 1; i + 1 < face.length; i++) {
      let b = vertices[face[i]];
      let c = vertices[face[i + 1]];
      let triN = cross3(sub3(b, a0), sub3(c, a0));
      if (dot3(triN, n) < 0) {
        const tmp = b;
        b = c;
        c = tmp;
        triN = cross3(sub3(b, a0), sub3(c, a0));
      }

      const pa = sub3(a0, ref);
      const pb = sub3(b, ref);
      const pc = sub3(c, ref);
      let vol = dot3(pa, cross3(pb, pc)) / 6;
      if (vol < 0) vol = -vol;
      if (vol <= 1e-15) continue;

      const tetraC = mul3(add3(add3(add3(ref, a0), b), c), 0.25);
      comNum[0] += tetraC[0] * vol;
      comNum[1] += tetraC[1] * vol;
      comNum[2] += tetraC[2] * vol;
      totalVol += vol;
    }
  }

  if (totalVol <= 1e-15) {
    return { volume: 0, centerOfMass: [ref[0], ref[1], ref[2]] };
  }
  return {
    volume: totalVol,
    centerOfMass: [comNum[0] / totalVol, comNum[1] / totalVol, comNum[2] / totalVol],
  };
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
    // Fallback: greedy with unbounded color count.
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

export function buildPolyDerivedCache(state: PolyState): PolyDerivedCache {
  const { vertices, faces, facePlanes } = state;
  const incidence = faces.map((face) => new Set<number>(face));

  let planarityMetric = 0;
  for (let fi = 0; fi < faces.length; fi++) {
    const plane = facePlanes[fi];
    for (let li = 0; li < faces[fi].length; li++) {
      const vi = faces[fi][li];
      const p = vertices[vi];
      const d = dot3(plane.n, p) - plane.b;
      planarityMetric += d * d;
    }
  }

  let unitNormalityMetric = 0;
  for (let fi = 0; fi < facePlanes.length; fi++) {
    const n = facePlanes[fi].n;
    const d = dot3(n, n) - 1;
    unitNormalityMetric += d * d;
  }

  let convexityViolation = 0;
  let isConvex = true;
  for (let fi = 0; fi < faces.length; fi++) {
    const plane = facePlanes[fi];
    for (let vi = 0; vi < vertices.length; vi++) {
      if (incidence[fi].has(vi)) continue;
      const val = dot3(plane.n, vertices[vi]) - plane.b;
      if (val > 0) {
        isConvex = false;
        convexityViolation += val * val;
      }
    }
  }

  const { volume, centerOfMass } = computeVolumeAndCom(state);
  const projectedComByFace: Vec3[] = facePlanes.map((pl) => projectPointToPlane(centerOfMass, pl));

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
    planarityMetric,
    unitNormalityMetric,
    convexityViolation,
    isConvex,
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
