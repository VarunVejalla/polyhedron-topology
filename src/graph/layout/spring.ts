import type { NodeId, SimpleGraph } from "../types";
import { cloneGraph } from "../core";

function clamp(v: number, a: number, b: number) {
  return Math.max(a, Math.min(b, v));
}

export function springLayoutStep(
  g: SimpleGraph,
  pinned: Set<NodeId>,
  canvas: { w: number; h: number; padding: number },
  stepScale = 1
): SimpleGraph {
  // Simple force-directed iteration (Fruchterman–Reingold-ish), bounded and stable.
  const { w, h, padding } = canvas;
  const n = g.nodes.length;
  if (n === 0) return g;

  const pos = new Map<NodeId, { x: number; y: number }>();
  for (const nd of g.nodes) pos.set(nd.id, { x: nd.x, y: nd.y });

  const k = Math.sqrt((w * h) / Math.max(1, n));
  const disp = new Map<NodeId, { dx: number; dy: number }>();
  for (const nd of g.nodes) disp.set(nd.id, { dx: 0, dy: 0 });

  // Repulsion
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = g.nodes[i];
      const b = g.nodes[j];
      const pa = pos.get(a.id)!;
      const pb = pos.get(b.id)!;
      const dx = pa.x - pb.x;
      const dy = pa.y - pb.y;
      const dist = Math.sqrt(dx * dx + dy * dy) + 1e-6;
      const f = (k * k) / dist;
      const ux = (dx / dist) * f;
      const uy = (dy / dist) * f;
      disp.get(a.id)!.dx += ux;
      disp.get(a.id)!.dy += uy;
      disp.get(b.id)!.dx -= ux;
      disp.get(b.id)!.dy -= uy;
    }
  }

  // Attraction along edges
  for (const e of g.edges) {
    const pa = pos.get(e.source);
    const pb = pos.get(e.target);
    if (!pa || !pb) continue;
    const dx = pa.x - pb.x;
    const dy = pa.y - pb.y;
    const dist = Math.sqrt(dx * dx + dy * dy) + 1e-6;
    const f = (dist * dist) / k;
    const ux = (dx / dist) * f;
    const uy = (dy / dist) * f;
    disp.get(e.source)!.dx -= ux;
    disp.get(e.source)!.dy -= uy;
    disp.get(e.target)!.dx += ux;
    disp.get(e.target)!.dy += uy;
  }

  // Integrate with a capped step.
  const maxStep = 10 * stepScale;
  const next = cloneGraph(g);
  for (let i = 0; i < next.nodes.length; i++) {
    const nd = next.nodes[i];
    if (pinned.has(nd.id)) continue;
    const d = disp.get(nd.id)!;
    const mag = Math.sqrt(d.dx * d.dx + d.dy * d.dy) + 1e-6;
    const sx = (d.dx / mag) * Math.min(maxStep, mag);
    const sy = (d.dy / mag) * Math.min(maxStep, mag);
    nd.x = clamp(nd.x + sx, padding, w - padding);
    nd.y = clamp(nd.y + sy, padding, h - padding);
  }
  return next;
}
