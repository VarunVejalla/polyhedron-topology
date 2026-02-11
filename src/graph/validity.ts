import type { PolyhedralCheck, SimpleGraph } from "./types";
import { isThreeVertexConnected } from "./core";
import { planarityWithEmbedding } from "./validity/planarityWithEmbedding";

/**
 * Polyhedral gate: planar + 3-vertex-connected + at least 4 vertices.
 *
 * This is intentionally the ONLY high-level validity contract the rest of the app relies on.
 * Editing/layout can still operate on arbitrary graphs; faces/dual/polyhedron must be hard-blocked unless ok=true.
 */
export function checkPolyhedral(g: SimpleGraph): PolyhedralCheck {
  if (g.nodes.length < 4) return { ok: false, reason: "too_small" };

  const pl = planarityWithEmbedding(g);
  if (!pl.ok) return { ok: false, reason: "nonplanar" };

  const three = isThreeVertexConnected(g);
  if (!three) return { ok: false, reason: "not_3_connected" };

  return { ok: true, embedding: pl.embedding };
}
