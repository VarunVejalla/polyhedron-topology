import { NodeId, RotationSystem, SimpleGraph } from "../types";

/**
 * Planarity test with embedding.
 *
 * Returns:
 *  - ok=true + a rotation system (cyclic neighbor order) when planar.
 *  - ok=false when nonplanar.
 *
 * Implementation: Brandes' Left-Right planarity test (2009), Algorithms 1–6.
 *
 * This is a *combinatorial* embedding; layout (Tutte, spring, manual) should be
 * treated separately.
 */
type PlanarityWithEmbeddingResult =
  | { ok: true; embedding: RotationSystem }
  | { ok: false };

export function planarityWithEmbedding(input: SimpleGraph): PlanarityWithEmbeddingResult {
  // Higher-level validity code already enforces simplicity/disconnected checks.
  // Keep this function purely focused on planarity + embedding.
  const g = input;

  // Trivial cases
  if (g.nodes.length <= 2) {
    const rot: RotationSystem = {};
    for (const n of g.nodes) {
      rot[n.id] = neighborsOf(g, n.id).sort();
    }
    return { ok: true, embedding: rot };
  }

  // Quick necessary condition
  if (g.edges.length > 3 * g.nodes.length - 6) {
    return { ok: false };
  }

  // ===== Build undirected adjacency + edge endpoints =====
  const nodeIds = g.nodes.map((n) => n.id);
  const adjUndir = new Map<NodeId, NodeId[]>();
  const endpoints = new Map<string, { a: NodeId; b: NodeId }>();
  for (const v of nodeIds) adjUndir.set(v, []);
  for (const e of g.edges) {
    endpoints.set(e.id, { a: e.source, b: e.target });
    adjUndir.get(e.source)!.push(e.target);
    adjUndir.get(e.target)!.push(e.source);
  }

  // ===== Oriented graph representation =====
  // We orient each undirected edge exactly once, producing a directed edge key.
  type EdgeKey = string; // `${edgeId}|${u}->${v}`
  const orientedOf = new Map<string, EdgeKey>(); // edgeId -> oriented key
  const srcOf = new Map<EdgeKey, NodeId>();
  const tgtOf = new Map<EdgeKey, NodeId>();
  const edgeIdOf = new Map<EdgeKey, string>();

  const height = new Map<NodeId, number>();
  const parentEdge = new Map<NodeId, EdgeKey | null>();

  const lowpt = new Map<EdgeKey, number>();
  const lowpt2 = new Map<EdgeKey, number>();
  const nestingDepth = new Map<EdgeKey, number>();

  const outgoing = new Map<NodeId, EdgeKey[]>();
  for (const v of nodeIds) {
    height.set(v, Number.POSITIVE_INFINITY);
    parentEdge.set(v, null);
    outgoing.set(v, []);
  }

  const makeEdgeKey = (edgeId: string, u: NodeId, v: NodeId) => `${edgeId}|${u}->${v}`;
  const other = (edgeId: string, u: NodeId): NodeId => {
    const ep = endpoints.get(edgeId);
    if (!ep) throw new Error(`Unknown edge ${edgeId}`);
    return ep.a === u ? ep.b : ep.a;
  };

  // Track whether an undirected edge is already oriented.
  const oriented = new Set<string>(); // edgeId

  // For stable iteration of edges, build per-node incident edge list.
  const incidentEdges = new Map<NodeId, string[]>();
  for (const v of nodeIds) incidentEdges.set(v, []);
  for (const e of g.edges) {
    incidentEdges.get(e.source)!.push(e.id);
    incidentEdges.get(e.target)!.push(e.id);
  }

  const roots: NodeId[] = [];

  // ===== Phase 1: DFS1 orientation + lowpoints + nesting depth (Alg. 2) =====
  const dfs1 = (v: NodeId) => {
    const eParent = parentEdge.get(v) ?? null;
    const inc = incidentEdges.get(v)!;

    for (const eid of inc) {
      if (oriented.has(eid)) continue;

      const w = other(eid, v);
      // orient {v,w} as (v,w)
      oriented.add(eid);
      const ek = makeEdgeKey(eid, v, w);
      orientedOf.set(eid, ek);
      srcOf.set(ek, v);
      tgtOf.set(ek, w);
      edgeIdOf.set(ek, eid);
      outgoing.get(v)!.push(ek);

      // init lowpoints
      lowpt.set(ek, height.get(v)!);
      lowpt2.set(ek, height.get(v)!);

      if (height.get(w)! === Number.POSITIVE_INFINITY) {
        // tree edge
        parentEdge.set(w, ek);
        height.set(w, height.get(v)! + 1);
        dfs1(w);
      } else {
        // back edge
        lowpt.set(ek, height.get(w)!);
      }

      // determine nesting depth
      let nd = 2 * lowpt.get(ek)!;
      if (lowpt2.get(ek)! < height.get(v)!) nd += 1; // chordal
      nestingDepth.set(ek, nd);

      // update lowpoints of parent edge e
      if (eParent !== null) {
        const lp = lowpt.get(ek)!;
        const lp2 = lowpt2.get(ek)!;
        const lpE = lowpt.get(eParent)!;
        const lp2E = lowpt2.get(eParent)!;
        if (lp < lpE) {
          lowpt2.set(eParent, Math.min(lpE, lp2));
          lowpt.set(eParent, lp);
        } else if (lp > lpE) {
          lowpt2.set(eParent, Math.min(lp2E, lp));
        } else {
          lowpt2.set(eParent, Math.min(lp2E, lp2));
        }
      }
    }
  };

  for (const s of nodeIds) {
    if (height.get(s)! !== Number.POSITIVE_INFINITY) continue;
    height.set(s, 0);
    roots.push(s);
    dfs1(s);
  }

  // ===== Phase 2: DFS2 testing (Alg. 3–5) =====
  // sort outgoing adjacency lists by non-decreasing nesting depth
  for (const v of nodeIds) {
    outgoing.get(v)!.sort((a, b) => (nestingDepth.get(a)! - nestingDepth.get(b)!));
  }

  type Interval = { high: EdgeKey | null; low: EdgeKey | null };
  type ConflictPair = { L: Interval; R: Interval };
  const emptyInterval = (): Interval => ({ high: null, low: null });
  const isEmptyInterval = (I: Interval) => I.high === null && I.low === null;
  const emptyPair = (): ConflictPair => ({ L: emptyInterval(), R: emptyInterval() });

  // ref/side arrays (edges of constraint forest)
  const ref = new Map<EdgeKey, EdgeKey | null>();
  const side = new Map<EdgeKey, 1 | -1>();
  for (const ek of srcOf.keys()) {
    ref.set(ek, null);
    side.set(ek, 1);
  }

  const stack: ConflictPair[] = [];
  const stackBottom = new Map<EdgeKey, number>();
  const lowptEdge = new Map<EdgeKey, EdgeKey>();

  const conflicting = (I: Interval, b: EdgeKey) => {
    if (isEmptyInterval(I)) return false;
    const high = I.high!;
    return lowpt.get(high)! > lowpt.get(b)!;
  };

  const lowest = (P: ConflictPair): number => {
    if (isEmptyInterval(P.L)) return lowpt.get(P.R.low!)!;
    if (isEmptyInterval(P.R)) return lowpt.get(P.L.low!)!;
    return Math.min(lowpt.get(P.L.low!)!, lowpt.get(P.R.low!)!);
  };

  const addConstraints = (e: EdgeKey, ei: EdgeKey) => {
    // Algorithm 4
    const P: ConflictPair = emptyPair();

    // merge return edges of ei into P.R
    const bottom = stackBottom.get(ei) ?? 0;
    while (stack.length > bottom) {
      const Q = stack.pop()!;

      // if Q.L != empty swap
      if (!isEmptyInterval(Q.L)) {
        const tmp = Q.L;
        Q.L = Q.R;
        Q.R = tmp;
      }
      if (!isEmptyInterval(Q.L)) {
        throw new Error("nonplanar");
      }

      // Q.R is non-empty
      if (lowpt.get(Q.R.low!)! > lowpt.get(e)!) {
        // merge intervals
        if (isEmptyInterval(P.R)) {
          P.R.high = Q.R.high;
        } else {
          ref.set(P.R.low!, Q.R.high!);
        }
        P.R.low = Q.R.low;
      } else {
        // align
        ref.set(Q.R.low!, lowptEdge.get(e)!);
      }

      // stop once we have consumed all pairs above stackBottom[ei]
    }

    // merge conflicting return edges of e1..ei-1 into P.L
    while (stack.length > 0 && (conflicting(stack[stack.length - 1].L, ei) || conflicting(stack[stack.length - 1].R, ei))) {
      const Q = stack.pop()!;
      if (conflicting(Q.R, ei)) {
        const tmp = Q.L;
        Q.L = Q.R;
        Q.R = tmp;
      }
      if (conflicting(Q.R, ei)) {
        throw new Error("nonplanar");
      }

      // merge interval below lowpt(ei) into P.R
      if (!isEmptyInterval(P.R)) {
        ref.set(P.R.low!, Q.R.high!);
        if (Q.R.low !== null) P.R.low = Q.R.low;
      } else {
        // P.R must exist by construction in Alg4, but guard.
        P.R.high = Q.R.high;
        P.R.low = Q.R.low;
      }

      if (isEmptyInterval(P.L)) {
        P.L.high = Q.L.high;
      } else {
        ref.set(P.L.low!, Q.L.high!);
      }
      P.L.low = Q.L.low;
    }

    if (!(isEmptyInterval(P.L) && isEmptyInterval(P.R))) stack.push(P);
  };

  const trimBackEdges = (e: EdgeKey) => {
    // Algorithm 5: remove back edges ending at parent u = source(e)
    const u = srcOf.get(e)!;
    // drop entire conflict pairs
    while (stack.length > 0 && lowest(stack[stack.length - 1]) === height.get(u)!) {
      const P = stack.pop()!;
      if (P.L.low !== null) side.set(P.L.low, -1);
    }
    if (stack.length === 0) return;

    // one more conflict pair to consider
    const P = stack.pop()!;

    // trim left interval
    while (P.L.high !== null && tgtOf.get(P.L.high)! === u) {
      P.L.high = ref.get(P.L.high) ?? null;
    }
    if (P.L.high === null && P.L.low !== null) {
      // just emptied
      ref.set(P.L.low, P.R.low);
      side.set(P.L.low, -1);
      P.L.low = null;
    }

    // trim right interval (symmetric)
    while (P.R.high !== null && tgtOf.get(P.R.high)! === u) {
      P.R.high = ref.get(P.R.high) ?? null;
    }
    if (P.R.high === null && P.R.low !== null) {
      ref.set(P.R.low, P.L.low);
      side.set(P.R.low, -1);
      P.R.low = null;
    }

    stack.push(P);
  };

  const dfs2 = (v: NodeId) => {
    const e = parentEdge.get(v) ?? null;
    const outs = outgoing.get(v)!;
    for (let i = 0; i < outs.length; i++) {
      const ei = outs[i];
      stackBottom.set(ei, stack.length);

      const w = tgtOf.get(ei)!;
      if (parentEdge.get(w) === ei) {
        // tree edge
        dfs2(w);
      } else {
        // back edge
        lowptEdge.set(ei, ei);
        stack.push({ L: emptyInterval(), R: { high: ei, low: ei } });
      }

      // integrate new return edges
      if (lowpt.get(ei)! < height.get(v)!) {
        if (i === 0) {
          if (e !== null) lowptEdge.set(e, lowptEdge.get(ei)!);
        } else {
          if (e === null) {
            // root: shouldn't happen with return edges
            continue;
          }
          addConstraints(e, ei);
        }
      }
    }

    if (e !== null) {
      const u = srcOf.get(e)!;
      trimBackEdges(e);

      // side of e is side of a highest return edge
      if (lowpt.get(e)! < height.get(u)!) {
        // e has return edge
        const top = stack[stack.length - 1];
        const hL = top?.L.high ?? null;
        const hR = top?.R.high ?? null;
        if (hL !== null && (hR === null || lowpt.get(hL)! > lowpt.get(hR)!)) {
          ref.set(e, hL);
        } else {
          ref.set(e, hR);
        }
      }
    }
  };

  try {
    for (const s of roots) dfs2(s);
  } catch {
    return { ok: false };
  }

  // ===== Phase 3: embedding =====
  // Dereference ref pointers to explicit side assignment (Algorithm 1: sign())
  const signMemo = new Map<EdgeKey, 1 | -1>();
  const sign = (ek: EdgeKey): 1 | -1 => {
    if (signMemo.has(ek)) return signMemo.get(ek)!;
    const r = ref.get(ek);
    let sgn: 1 | -1 = side.get(ek)!;
    if (r) {
      sgn = (sgn * sign(r)) as 1 | -1;
      ref.set(ek, null);
      side.set(ek, sgn);
    }
    signMemo.set(ek, sgn);
    return sgn;
  };
  for (const ek of srcOf.keys()) sign(ek);

  // Multiply nestingDepth by sign and re-sort outgoing adjacency lists
  for (const ek of srcOf.keys()) {
    nestingDepth.set(ek, (nestingDepth.get(ek)! * side.get(ek)!) | 0);
  }
  for (const v of nodeIds) {
    outgoing.get(v)!.sort((a, b) => (nestingDepth.get(a)! - nestingDepth.get(b)!));
  }

  // Rotation structure per vertex: store incident undirected edge IDs in cyclic order.
  // IMPORTANT: Do NOT seed with an arbitrary incident-edge order. Algorithm 6 constructs
  // the cyclic order incrementally via local insertions. Seeding with an arbitrary list
  // and then "moving" edges around can leave the final rotation inconsistent, which
  // later breaks face-walking (non-simple face cycles).
  const rotEdges = new Map<NodeId, string[]>();
  const leftRef = new Map<NodeId, string | null>();
  const rightRef = new Map<NodeId, string | null>();
  for (const v of nodeIds) {
    rotEdges.set(v, []);
    leftRef.set(v, null);
    rightRef.set(v, null);
  }

  const moveEdgeToFront = (v: NodeId, eid: string) => {
    const arr = rotEdges.get(v)!;
    const idx = arr.indexOf(eid);
    if (idx >= 0) arr.splice(idx, 1);
    arr.unshift(eid);
  };
  const insertAfter = (v: NodeId, refEid: string, eid: string) => {
    const arr = rotEdges.get(v)!;
    const curIdx = arr.indexOf(eid);
    if (curIdx >= 0) arr.splice(curIdx, 1);
    const idx = arr.indexOf(refEid);
    const ins = idx >= 0 ? idx + 1 : arr.length;
    arr.splice(ins, 0, eid);
  };
  const insertBefore = (v: NodeId, refEid: string, eid: string) => {
    const arr = rotEdges.get(v)!;
    const curIdx = arr.indexOf(eid);
    if (curIdx >= 0) arr.splice(curIdx, 1);
    const idx = arr.indexOf(refEid);
    const ins = idx >= 0 ? idx : 0;
    arr.splice(ins, 0, eid);
  };

  const ensureInList = (v: NodeId, eid: string) => {
    const arr = rotEdges.get(v)!;
    if (!arr.includes(eid)) arr.push(eid);
  };

  const dfs3 = (v: NodeId) => {
    const outs = outgoing.get(v)!;
    for (const ei of outs) {
      const w = tgtOf.get(ei)!;
      const eid = edgeIdOf.get(ei)!;

      // This edge is incident to BOTH endpoints; make sure it appears in both lists.
      // The insertion logic below primarily orders it around the target endpoint w
      // (matching Brandes Alg. 6). For the source endpoint v, we keep a stable
      // order consistent with the DFS1/LR ordering by appending in outgoing order.
      ensureInList(v, eid);
      ensureInList(w, eid);

      if (parentEdge.get(w) === ei) {
        // tree edge (v -> w): make it first in adjacency list of w
        moveEdgeToFront(w, eid);

        // IMPORTANT (Brandes Alg. 6): refs are stored on *v*, not w.
        // They mark the tree edge leading into the currently processed subtree.
        leftRef.set(v, eid);
        rightRef.set(v, eid);

        dfs3(w);
      } else {
        // back edge (v -> w): insert into adjacency list of w based on side
        if (side.get(ei)! === 1) {
          const rr = rightRef.get(w);
          if (rr) insertAfter(w, rr, eid);
          else rotEdges.get(w)!.push(eid);
        } else {
          const lr = leftRef.get(w);
          if (lr) {
            insertBefore(w, lr, eid);
            leftRef.set(w, eid);
          } else {
            rotEdges.get(w)!.unshift(eid);
            leftRef.set(w, eid);
          }
        }
      }
    }
  };

  for (const s of roots) dfs3(s);

  // (rotEdges are already seeded for all vertices.)

  // Convert edge-rotation to neighbor-rotation.
  const rotation: RotationSystem = {};
  for (const v of nodeIds) {
    const orderEdges = rotEdges.get(v)!;
    // Remove duplicates while preserving order (can happen due to guards above).
    const seen = new Set<string>();
    const dedupEdges: string[] = [];
    for (const eid of orderEdges) {
      if (seen.has(eid)) continue;
      seen.add(eid);
      dedupEdges.push(eid);
    }
    // Map to neighbors
    const neigh = dedupEdges.map((eid) => other(eid, v));
    // Additional robustness: remove any undefined/null neighbors that might have slipped through
    const validNeigh = neigh.filter((u) => u !== undefined && u !== null);
    rotation[v] = validNeigh;
  }

  // Sanity: every vertex must list all its neighbors exactly once, and the rotation must be bidirectional.
  // If this fails, it indicates a bug in the embedding-construction phase. Since the planarity test
  // has already succeeded, we treat this as an internal error (callers requested a reliable embedding).
  for (const v of nodeIds) {
    const expected = new Set(adjUndir.get(v)!);
    const got = rotation[v];

    if (got.length !== expected.size) {
      throw new Error("planarityWithEmbedding: embedding construction failed sanity (degree mismatch)");
    }
    const uniq = new Set(got);
    if (uniq.size !== got.length) {
      throw new Error("planarityWithEmbedding: embedding construction failed sanity (duplicate neighbor)");
    }
    for (const u of got) {
      if (!expected.has(u)) {
        throw new Error("planarityWithEmbedding: embedding construction failed sanity (bad neighbor)");
      }
      if (!(rotation[u] ?? []).includes(v)) {
        throw new Error("planarityWithEmbedding: embedding construction failed sanity (non-bidirectional)");
      }
    }
  }

  return { ok: true, embedding: rotation };
}

function neighborsOf(g: SimpleGraph, v: NodeId): NodeId[] {
  const out: NodeId[] = [];
  for (const e of g.edges) {
    if (e.source === v) out.push(e.target);
    else if (e.target === v) out.push(e.source);
  }
  return out;
}
