import type { Vec3 } from "../math/types";
import { v3 } from "../math/vec3";
import { EPS, POLYGON_INSIDE_EPS } from "../math/constants";
import { buildPolyRichState } from "./auxiliary";
import type { PolyDerivedCache, PolyRichState, PolyState, RollStep } from "./types";

function edgeKey(a: number, b: number): string {
  return a < b ? `${a},${b}` : `${b},${a}`;
}

function centroidOfFace(vertices: ReadonlyArray<Vec3>, face: ReadonlyArray<number>): Vec3 {
  const c: Vec3 = [0, 0, 0];
  if (face.length === 0) return c;
  for (let i = 0; i < face.length; i++) {
    const p = vertices[face[i]];
    c[0] += p[0];
    c[1] += p[1];
    c[2] += p[2];
  }
  const inv = 1 / face.length;
  return [c[0] * inv, c[1] * inv, c[2] * inv];
}

function computeConstraintMetrics(state: PolyState) {
  const faceSets = state.faces.map((f) => new Set<number>(f));
  let planarityMetric = 0;
  let unitNormalityMetric = 0;
  let convexityViolation = 0;
  let isConvex = true;

  for (let fi = 0; fi < state.faces.length; fi++) {
    const face = state.faces[fi];
    const plane = state.facePlanes[fi];
    for (let i = 0; i < face.length; i++) {
      const p = state.vertices[face[i]];
      const d = v3.dot(plane.n, p) - plane.b;
      planarityMetric += d * d;
    }
    const unit = v3.dot(plane.n, plane.n) - 1;
    unitNormalityMetric += unit * unit;
    for (let vi = 0; vi < state.vertices.length; vi++) {
      if (faceSets[fi].has(vi)) continue;
      const out = v3.dot(plane.n, state.vertices[vi]) - plane.b;
      if (out > 0) {
        isConvex = false;
        convexityViolation += out * out;
      }
    }
  }

  return { planarityMetric, unitNormalityMetric, convexityViolation, isConvex };
}

function polygonMargin(
  vertices: ReadonlyArray<Vec3>,
  face: ReadonlyArray<number>,
  n: Readonly<Vec3>,
  q: Readonly<Vec3>
): { margin: number; inside: boolean; minEdgeIndex: number } {
  if (face.length < 3) return { margin: -Infinity, inside: false, minEdgeIndex: -1 };
  const center = centroidOfFace(vertices, face);

  let orientSign = 1;
  for (let i = 0; i < face.length; i++) {
    const a = vertices[face[i]];
    const b = vertices[face[(i + 1) % face.length]];
    const s = v3.dot(v3.cross(v3.sub(b, a), v3.sub(center, a)), n);
    if (Math.abs(s) > EPS) {
      orientSign = s >= 0 ? 1 : -1;
      break;
    }
  }

  let minMargin = Number.POSITIVE_INFINITY;
  let minEdgeIndex = -1;
  for (let i = 0; i < face.length; i++) {
    const a = vertices[face[i]];
    const b = vertices[face[(i + 1) % face.length]];
    const edge = v3.sub(b, a);
    const len = Math.max(EPS, v3.norm(edge));
    const s = v3.dot(v3.cross(edge, v3.sub(q, a)), n) / len;
    const margin = s * orientSign;
    if (margin < minMargin) {
      minMargin = margin;
      minEdgeIndex = i;
    }
  }

  return { margin: minMargin, inside: minMargin >= -POLYGON_INSIDE_EPS, minEdgeIndex };
}

function buildFaceAdjacency(faces: ReadonlyArray<ReadonlyArray<number>>) {
  const byEdge = new Map<string, number[]>();
  const neighbors = Array.from({ length: faces.length }, () => new Set<number>());

  for (let fi = 0; fi < faces.length; fi++) {
    const face = faces[fi];
    for (let i = 0; i < face.length; i++) {
      const key = edgeKey(face[i], face[(i + 1) % face.length]);
      const owners = byEdge.get(key);
      if (owners) owners.push(fi);
      else byEdge.set(key, [fi]);
    }
  }

  for (const owners of byEdge.values()) {
    if (owners.length !== 2) continue;
    neighbors[owners[0]].add(owners[1]);
    neighbors[owners[1]].add(owners[0]);
  }

  return { byEdge, neighbors };
}

function sixColorPlanar(nodes: number[], neighbors: Map<number, Set<number>>): Map<number, number> {
  const degree = new Map<number, number>();
  const removed = new Set<number>();
  const stack: number[] = [];
  const buckets: Array<Set<number>> = Array.from({ length: 6 }, () => new Set<number>());

  for (let i = 0; i < nodes.length; i++) {
    const v = nodes[i];
    const d = neighbors.get(v)?.size ?? 0;
    degree.set(v, d);
    buckets[Math.min(5, d)].add(v);
  }

  for (let k = 0; k < nodes.length; k++) {
    let v: number | null = null;
    for (let d = 0; d <= 5; d++) {
      const it = buckets[d].values().next();
      if (!it.done) {
        v = it.value;
        buckets[d].delete(v);
        break;
      }
    }
    if (v == null) throw new Error("sixColorPlanar failed: no vertex with degree <= 5");

    removed.add(v);
    stack.push(v);
    for (const u of neighbors.get(v) ?? []) {
      if (removed.has(u)) continue;
      const oldD = degree.get(u) ?? 0;
      buckets[Math.min(5, oldD)].delete(u);
      const nextD = Math.max(0, oldD - 1);
      degree.set(u, nextD);
      buckets[Math.min(5, nextD)].add(u);
    }
  }

  const color = new Map<number, number>();
  const colorUseCount = new Array<number>(6).fill(0);
  while (stack.length > 0) {
    const v = stack.pop() as number;
    const used = new Array<boolean>(6).fill(false);
    for (const u of neighbors.get(v) ?? []) {
      const c = color.get(u);
      if (c !== undefined && c >= 0 && c < 6) used[c] = true;
    }
    let assigned = -1;
    let bestCount = Number.POSITIVE_INFINITY;
    for (let c = 0; c < 6; c++) {
      if (!used[c] && colorUseCount[c] < bestCount) {
        assigned = c;
        bestCount = colorUseCount[c];
      }
    }
    if (assigned < 0) throw new Error("sixColorPlanar failed: no available color in 0..5");
    color.set(v, assigned);
    colorUseCount[assigned] += 1;
  }
  return color;
}

function isRichState(state: PolyState | PolyRichState): state is PolyRichState {
  return "topology" in state && "aux" in state;
}

export function buildPolyDerivedCache(stateArg: PolyState | PolyRichState): PolyDerivedCache {
  const rich = isRichState(stateArg) ? stateArg : buildPolyRichState(stateArg);
  const { vertices, faces, facePlanes, aux } = rich;
  const metrics = computeConstraintMetrics(rich);

  const faceNormals = facePlanes.map((pl) => [pl.n[0], pl.n[1], pl.n[2]] as Vec3);
  const faceCentroids = aux.faceCentroid.map((c) => [c[0], c[1], c[2]] as Vec3);
  const centerOfMass: Vec3 = [aux.centerOfMass[0], aux.centerOfMass[1], aux.centerOfMass[2]];
  const projectedComByFace = aux.projectedComByFace.map((p) => [p[0], p[1], p[2]] as Vec3);

  const { byEdge, neighbors } = buildFaceAdjacency(faces);

  const stableFace = new Array<boolean>(faces.length);
  const signedMarginByFace = new Array<number>(faces.length);
  const nextRollByFace: RollStep[] = new Array(faces.length);

  for (let fi = 0; fi < faces.length; fi++) {
    const margin = polygonMargin(vertices, faces[fi], facePlanes[fi].n, projectedComByFace[fi]);
    stableFace[fi] = margin.inside;
    signedMarginByFace[fi] = margin.margin;
    if (margin.inside || margin.minEdgeIndex < 0) {
      nextRollByFace[fi] = { nextFace: null, edge: null };
      continue;
    }
    const a = faces[fi][margin.minEdgeIndex];
    const b = faces[fi][(margin.minEdgeIndex + 1) % faces[fi].length];
    const owners = byEdge.get(edgeKey(a, b)) ?? [];
    const nextFace = owners.length >= 2 ? (owners[0] === fi ? owners[1] : owners[0]) : null;
    nextRollByFace[fi] = { nextFace, edge: [a, b] };
  }

  const settleFaceByFace: Array<number | null> = new Array(faces.length).fill(null);
  const settlePathByFace: number[][] = new Array(faces.length);
  for (let fi = 0; fi < faces.length; fi++) {
    const path: number[] = [fi];
    const visited = new Set<number>([fi]);
    let cur = fi;
    while (true) {
      if (stableFace[cur]) {
        settleFaceByFace[fi] = cur;
        break;
      }
      const next = nextRollByFace[cur].nextFace;
      if (next == null || visited.has(next)) {
        if (next != null) path.push(next);
        break;
      }
      path.push(next);
      visited.add(next);
      cur = next;
    }
    settlePathByFace[fi] = path;
  }

  const faceIds = faces.map((_f, i) => i);
  const faceAdj = new Map<number, Set<number>>();
  for (let fi = 0; fi < faces.length; fi++) faceAdj.set(fi, neighbors[fi]);
  const defaultColor = sixColorPlanar(faceIds, faceAdj);
  const defaultColorByFace = faceIds.map((fi) => defaultColor.get(fi) ?? 0);

  const basinIdByFace = new Array<number>(faces.length);
  const basinFaces = new Map<number, number[]>();
  for (let fi = 0; fi < faces.length; fi++) {
    const basin = settleFaceByFace[fi] ?? -(fi + 1);
    basinIdByFace[fi] = basin;
    const bucket = basinFaces.get(basin);
    if (bucket) bucket.push(fi);
    else basinFaces.set(basin, [fi]);
  }

  const basinAdj = new Map<number, Set<number>>();
  for (const basin of basinFaces.keys()) basinAdj.set(basin, new Set<number>());
  for (let fi = 0; fi < faces.length; fi++) {
    const bi = basinIdByFace[fi];
    for (const fj of neighbors[fi]) {
      const bj = basinIdByFace[fj];
      if (bi === bj) continue;
      basinAdj.get(bi)?.add(bj);
      basinAdj.get(bj)?.add(bi);
    }
  }
  const basinColor = sixColorPlanar([...basinFaces.keys()], basinAdj);
  const basinColorByFace = basinIdByFace.map((basin) => basinColor.get(basin) ?? 0);

  return {
    planarityMetric: metrics.planarityMetric,
    unitNormalityMetric: metrics.unitNormalityMetric,
    convexityViolation: metrics.convexityViolation,
    isConvex: metrics.isConvex,
    faceNormals,
    faceCentroids,
    centerOfMass,
    volume: aux.volume,
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
