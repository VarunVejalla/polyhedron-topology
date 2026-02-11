import type { GraphEdge, NodeId, RotationSystem, SimpleGraph } from "./types";

/** Compare two cycles up to rotation and reversal. */
function sameCycleUpToRotation(a: NodeId[], b: NodeId[]): boolean {
  if (a.length !== b.length) return false;
  const n = a.length;
  const idxs: number[] = [];
  for (let i = 0; i < n; i++) if (b[i] === a[0]) idxs.push(i);
  if (idxs.length === 0) return false;

  for (const s of idxs) {
    let okFwd = true;
    for (let k = 0; k < n; k++) {
      if (a[k] !== b[(s + k) % n]) {
        okFwd = false;
        break;
      }
    }
    if (okFwd) return true;

    let okRev = true;
    for (let k = 0; k < n; k++) {
      const j = (s - k + n) % n;
      if (a[k] !== b[j]) {
        okRev = false;
        break;
      }
    }
    if (okRev) return true;
  }
  return false;
}

export type Face = { id: string; cycle: NodeId[] };

type Dart = { u: NodeId; v: NodeId }; // directed edge u->v
const dkey = (d: Dart) => `${d.u}__${d.v}`;

function undirectedKey(a: NodeId, b: NodeId): string {
  return a < b ? `${a}__${b}` : `${b}__${a}`;
}

function predecessor(rot: RotationSystem, v: NodeId, u: NodeId): NodeId | null {
  const nbrs = rot[v] ?? [];
  const i = nbrs.indexOf(u);
  if (i < 0) return null;
  return nbrs[(i - 1 + nbrs.length) % nbrs.length];
}

/** Walk all faces from a rotation system (combinatorial embedding). */
export function facesFromEmbedding(nodes: NodeId[], edges: GraphEdge[], rot: RotationSystem): Face[] {
  // Enumerate faces by walking darts with a visited-dart set.
  // No "magic iteration cap": we visit each dart at most once (2m total),
  // and throw loudly if the rotation/predecessor relation is inconsistent.

  const nodeSet = new Set(nodes);
  const darts: Dart[] = [];
  for (const e of edges) {
    if (e.source === e.target) continue;
    if (!nodeSet.has(e.source) || !nodeSet.has(e.target)) {
      throw new Error(`facesFromEmbedding: edge ${e.source}-${e.target} references unknown node`);
    }
    darts.push({ u: e.source, v: e.target });
    darts.push({ u: e.target, v: e.source });
  }

  const used = new Set<string>();
  const faces: Face[] = [];

  for (const d0 of darts) {
    const k0 = dkey(d0);
    if (used.has(k0)) continue;

    const cycle: NodeId[] = [];
    let cur: Dart = d0;

    while (true) {
      const k = dkey(cur);
      if (used.has(k)) break;
      used.add(k);
      cycle.push(cur.u);

      const w = predecessor(rot, cur.v, cur.u);
      if (!w) throw new Error(`facesFromEmbedding: predecessor missing at v=${cur.v} from u=${cur.u}`);
      cur = { u: cur.v, v: w };

      if (dkey(cur) === k0) break;
    }

    if (cycle.length >= 3) {
      faces.push({ id: `f_${faces.length}`, cycle });
    }
  }

  // Filter non-simple cycles and deduplicate up to rotation.
  const simple: Face[] = [];
  for (const f of faces) {
    const seen = new Set<NodeId>();
    let ok = true;
    for (const v of f.cycle) {
      if (seen.has(v)) {
        ok = false;
        break;
      }
      seen.add(v);
    }
    if (ok) simple.push(f);
  }

  const uniq: Face[] = [];
  for (const f of simple) {
    if (uniq.some((g) => sameCycleUpToRotation(f.cycle, g.cycle))) continue;
    uniq.push(f);
  }

  return uniq;
}

export function chooseOuterFace(faces: Face[]): Face | null {
  if (!faces.length) return null;
  return [...faces].sort((a, b) => b.cycle.length - a.cycle.length)[0];
}

export function planarDualFromFaces(faces: Face[]): SimpleGraph {
  // Dual: one node per face; edges between faces that share an undirected edge.
  const edgeToFaces = new Map<string, string[]>();
  for (const f of faces) {
    const cyc = f.cycle;
    for (let i = 0; i < cyc.length; i++) {
      const a = cyc[i];
      const b = cyc[(i + 1) % cyc.length];
      const key = undirectedKey(a, b);
      const arr = edgeToFaces.get(key) ?? [];
      arr.push(f.id);
      edgeToFaces.set(key, arr);
    }
  }

  const nodes = faces.map((f, i) => ({ id: f.id, label: `F${i}`, x: 0, y: 0 }));
  const idSet = new Set(nodes.map((n) => n.id));
  const edges: { id: string; source: string; target: string }[] = [];
  let k = 0;
  const seen = new Set<string>();
  for (const fs of edgeToFaces.values()) {
    if (fs.length < 2) continue;
    const a = fs[0], b = fs[1];
    if (!idSet.has(a) || !idSet.has(b) || a === b) continue;
    const key = undirectedKey(a, b);
    if (seen.has(key)) continue;
    seen.add(key);
    edges.push({ id: `d_${k++}`, source: a, target: b });
  }

  return { nodes, edges };
}
