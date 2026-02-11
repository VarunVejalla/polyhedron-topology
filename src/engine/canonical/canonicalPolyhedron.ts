import type { SimpleGraph, NodeId } from "../../graph/types";
import type { Vec3 } from "../math/types";
import { v3 } from "../math/vec3";
import { EPS_AREA } from "../math/constants";
import { bestFitPlanePCA, planarityResiduals, projectPointToPlane } from "../geom/plane";


type CanonicalBuildInput = {
  vertexGraph: SimpleGraph; // planar embedded (x,y used only for initialization)
  faces: Array<{ id: string; cycle: NodeId[] }>; // includes the outside face
};

type CanonicalBuildOptions = {
  maxIters?: number;
  tol?: number;
  edgeStep?: number; // tangency correction step size
  planeBlend?: number; // 0..1 blend for planar projection aggregation
  initialScale?: number; // scale of initial lifted embedding
};

type CanonicalBuildResult = {
  vertices: Vec3[];
  faces: number[][]; // polygon vertex indices
  stats: {
    iters: number;
    maxEdgeTangencyError: number;
    centroidNorm: number;
    maxFacePlanarityRms: number;
  };
};

function stereographicInverse(u: number, v: number): Vec3 {
  const d = u * u + v * v + 1;
  return [
    (2 * u) / d,
    (2 * v) / d,
    (u * u + v * v - 1) / d,
  ];
}

function closestPointToOriginOnSegment(a: Vec3, b: Vec3): { x: Vec3; t: number } {
  const ab = v3.sub(b, a);
  const denom = v3.dot(ab, ab);
  if (denom < EPS_AREA) return { x: [...a] as Vec3, t: 0 };
  // minimize ||a + t(b-a)||^2 => t = -(a·(b-a))/||b-a||^2
  const t = v3.clamp01(-v3.dot(a, ab) / denom);
  return { x: v3.add(a, v3.mul(ab, t)), t };
}

/**
 * Numerical canonicalization iteration matching the user-provided description:
 *  1) edge tangency to unit sphere via closest-point correction
 *  2) centroid of tangency points to origin
 *  3) face planarity via quick plane approx + projection
 */
export function buildCanonicalPolyhedron(input: CanonicalBuildInput, options: CanonicalBuildOptions = {}): CanonicalBuildResult {
  const maxIters = options.maxIters ?? 400;
  const tol = options.tol ?? 1e-7;
  const edgeStep = options.edgeStep ?? 0.05;
  const planeBlend = options.planeBlend ?? 1.0;
  const initialScale = options.initialScale ?? 2.3;

  // Map node ids to indices
  const idToIndex = new Map<NodeId, number>();
  input.vertexGraph.nodes.forEach((n, i) => idToIndex.set(n.id, i));

  // Faces as index cycles
  const faces: number[][] = input.faces
    .map((f) => f.cycle.map((id) => idToIndex.get(id)).filter((x): x is number => typeof x === "number"))
    .filter((cyc) => cyc.length >= 3);

  // For deterministic plane-normal orientation across iterations.
  const prevFaceNormals: Array<Vec3 | undefined> = faces.map(() => undefined);

  // Edges list as index pairs
  const edges: Array<[number, number]> = input.vertexGraph.edges
    .map((e) => {
      const a = idToIndex.get(e.source);
      const b = idToIndex.get(e.target);
      return typeof a === "number" && typeof b === "number" ? ([a, b] as [number, number]) : null;
    })
    .filter((x): x is [number, number] => x !== null);

  // Initial vertex positions: lift embedded 2D to unit sphere via inverse stereographic then scale outward.
  const nodes = input.vertexGraph.nodes;
  const cx = nodes.reduce((s, n) => s + n.x, 0) / Math.max(1, nodes.length);
  const cy = nodes.reduce((s, n) => s + n.y, 0) / Math.max(1, nodes.length);
  let maxR = 1;
  for (const n of nodes) {
    const dx = n.x - cx;
    const dy = n.y - cy;
    maxR = Math.max(maxR, Math.sqrt(dx * dx + dy * dy));
  }
  const scale2D = maxR > 1e-9 ? (1 / maxR) : 1;
  const V: Vec3[] = nodes.map((n) => {
    const u = (n.x - cx) * scale2D;
    const v = (n.y - cy) * scale2D;
    const s = stereographicInverse(u, v);
    return v3.mul(s, initialScale);
  });

  let it = 0;
  let maxEdgeErr = Infinity;
  let centroidNorm = Infinity;
  let maxPlanarity = Infinity;

  // Iterations
  for (it = 0; it < maxIters; it++) {
    // --- (1) Edge tangency correction
    const tangencyPoints: Vec3[] = [];
    maxEdgeErr = 0;
    for (const [i, j] of edges) {
      const a = V[i];
      const b = V[j];
      const { x } = closestPointToOriginOnSegment(a, b);
      const r = v3.norm(x);
      tangencyPoints.push(r > 1e-12 ? v3.mul(x, 1 / r) : [0, 0, 1]);

      const err = 1 - r; // positive if closest point inside unit sphere
      maxEdgeErr = Math.max(maxEdgeErr, Math.abs(err));

      if (Math.abs(err) > 0) {
        // push/pull along radial direction of x
        const dir = r > 1e-12 ? v3.mul(x, 1 / r) : ([0, 0, 1] as Vec3);
        const delta = v3.mul(dir, edgeStep * err);
        V[i] = v3.add(V[i], delta);
        V[j] = v3.add(V[j], delta);
      }
    }

    // --- (2) Center tangency centroid
    let c: Vec3 = [0, 0, 0];
    for (const p of tangencyPoints) c = v3.add(c, p);
    c = v3.mul(c, 1 / Math.max(1, tangencyPoints.length));
    centroidNorm = v3.norm(c);
    if (centroidNorm > 0) {
      for (let i = 0; i < V.length; i++) V[i] = v3.sub(V[i], c);
    }

    // --- (3) Face planarity projection
    // Accumulate per-vertex projected positions from all incident faces, then average.
    const acc: Vec3[] = Array.from({ length: V.length }, () => [0, 0, 0] as Vec3);
    const cnt: number[] = Array.from({ length: V.length }, () => 0);
    maxPlanarity = 0;
    for (let fi = 0; fi < faces.length; fi++) {
      const cyc = faces[fi];
      const pts = cyc.map((vi) => V[vi]);
      const plane = bestFitPlanePCA(pts, prevFaceNormals[fi]);
      prevFaceNormals[fi] = plane.n;

      maxPlanarity = Math.max(maxPlanarity, planarityResiduals(pts, plane).rms);

      for (const vi of cyc) {
        const pproj = projectPointToPlane(V[vi], plane);
        acc[vi] = v3.add(acc[vi], pproj);
        cnt[vi] += 1;
      }
    }
    for (let i = 0; i < V.length; i++) {
      if (cnt[i] > 0) {
        const avg = v3.mul(acc[i], 1 / cnt[i]);
        // optional blend to reduce oscillation
        V[i] = v3.add(v3.mul(V[i], 1 - planeBlend), v3.mul(avg, planeBlend));
      }
    }

    // stopping
    if (maxEdgeErr < tol && centroidNorm < tol && maxPlanarity < tol) break;
  }

  return {
    vertices: V,
    faces,
    stats: {
      iters: it + 1,
      maxEdgeTangencyError: maxEdgeErr,
      centroidNorm,
      maxFacePlanarityRms: maxPlanarity,
    },
  };
}
