import { readLightNormal, readLightOffset, readLightVertex } from "./light";
import type { VertexFaceIncidence } from "./types";

function dot3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function incidenceConstraintValue(
  y: ReadonlyArray<number>,
  vertexCount: number,
  pair: VertexFaceIncidence
): number {
  const v = readLightVertex(y, pair.vi);
  const n = readLightNormal(y, vertexCount, pair.fi);
  const b = readLightOffset(y, vertexCount, pair.fi);
  return dot3(n, v) - b;
}

export function nonIncidenceConstraintValue(
  y: ReadonlyArray<number>,
  vertexCount: number,
  pair: VertexFaceIncidence,
  margin = 0
): number {
  return incidenceConstraintValue(y, vertexCount, pair) + margin;
}
