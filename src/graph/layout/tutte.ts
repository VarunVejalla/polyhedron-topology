import type { NodeId, SimpleGraph } from "../types";
import { buildAdjacency } from "../core";

/** Tutte barycentric straight-line drawing given a boundary cycle. */
export function tutteLayout(
  g: SimpleGraph,
  outer: NodeId[],
  canvas: { w: number; h: number; padding: number },
  pinned: Set<NodeId> = new Set()
): Map<NodeId, { x: number; y: number }> {
  const n = g.nodes.length;
  const idx = new Map<NodeId, number>();
  g.nodes.forEach((nd, i) => idx.set(nd.id, i));

  const boundary = outer.filter((v) => idx.has(v));
  const fixed = new Set<NodeId>([...boundary, ...pinned]);

  // place boundary on a circle
  const R = Math.min(canvas.w, canvas.h) * 0.42;
  const cx = canvas.w / 2;
  const cy = canvas.h / 2;

  const X = new Array(n).fill(0);
  const Y = new Array(n).fill(0);

  for (let i = 0; i < boundary.length; i++) {
    const v = boundary[i];
    const a = (2 * Math.PI * i) / boundary.length;
    const j = idx.get(v)!;
    X[j] = cx + R * Math.cos(a);
    Y[j] = cy + R * Math.sin(a);
  }
  for (const v of pinned) {
    const j = idx.get(v);
    if (j == null) continue;
    const node = g.nodes[j];
    X[j] = node.x;
    Y[j] = node.y;
  }

  const adj = buildAdjacency(g);
  const interior = g.nodes.map((nd) => nd.id).filter((v) => !fixed.has(v));

  for (let iter = 0; iter < 800; iter++) {
    let maxDelta = 0;
    for (const v of interior) {
      const j = idx.get(v)!;
      const nbrs = [...(adj.get(v) ?? [])];
      if (nbrs.length === 0) continue;
      let sx = 0, sy = 0;
      for (const u of nbrs) {
        const iu = idx.get(u);
        if (iu == null) continue;
        sx += X[iu];
        sy += Y[iu];
      }
      const nx = sx / nbrs.length;
      const ny = sy / nbrs.length;
      const dx = nx - X[j];
      const dy = ny - Y[j];
      X[j] = nx;
      Y[j] = ny;
      const d = Math.hypot(dx, dy);
      if (d > maxDelta) maxDelta = d;
    }
    if (maxDelta < 1e-3) break;
  }

  // clamp to canvas with padding
  let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
  for (let i = 0; i < n; i++) {
    minx = Math.min(minx, X[i]); maxx = Math.max(maxx, X[i]);
    miny = Math.min(miny, Y[i]); maxy = Math.max(maxy, Y[i]);
  }
  const pad = canvas.padding;
  const sx = (canvas.w - 2 * pad) / Math.max(1e-6, maxx - minx);
  const sy = (canvas.h - 2 * pad) / Math.max(1e-6, maxy - miny);
  const s = Math.min(sx, sy);

  const out = new Map<NodeId, { x: number; y: number }>();
  for (let i = 0; i < n; i++) {
    const v = g.nodes[i].id;
    out.set(v, { x: pad + (X[i] - minx) * s, y: pad + (Y[i] - miny) * s });
  }
  return out;
}
