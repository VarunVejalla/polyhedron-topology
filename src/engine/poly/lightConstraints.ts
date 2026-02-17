import type { Vec3 } from "../math/types";
import { readLightNormal, readLightOffset, readLightVertex } from "./light";
import type { VertexFaceIncidence } from "./types";

function dot3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

type LightIncidenceLinearization = {
  value: number;
  gV: Vec3;
  gN: Vec3;
  gB: -1;
};

type LightUnitNormalLinearization = {
  value: number;
  gN: Vec3;
};

type LightSquaredSlackLinearization = {
  value: number;
  gV: Vec3;
  gN: Vec3;
  gB: -1;
  gD: number;
};

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

export function incidenceConstraintLinearization(
  y: ReadonlyArray<number>,
  vertexCount: number,
  pair: VertexFaceIncidence
): LightIncidenceLinearization {
  const v = readLightVertex(y, pair.vi);
  const n = readLightNormal(y, vertexCount, pair.fi);
  return {
    value: dot3(n, v) - readLightOffset(y, vertexCount, pair.fi),
    gV: [n[0], n[1], n[2]],
    gN: [v[0], v[1], v[2]],
    gB: -1,
  };
}

export function unitNormalConstraintValue(
  y: ReadonlyArray<number>,
  vertexCount: number,
  fi: number
): number {
  const n = readLightNormal(y, vertexCount, fi);
  return dot3(n, n) - 1;
}

export function unitNormalConstraintLinearization(
  y: ReadonlyArray<number>,
  vertexCount: number,
  fi: number
): LightUnitNormalLinearization {
  const n = readLightNormal(y, vertexCount, fi);
  return {
    value: dot3(n, n) - 1,
    gN: [2 * n[0], 2 * n[1], 2 * n[2]],
  };
}

export function nonIncidenceConstraintValue(
  y: ReadonlyArray<number>,
  vertexCount: number,
  pair: VertexFaceIncidence,
  margin = 0
): number {
  return incidenceConstraintValue(y, vertexCount, pair) + margin;
}

export function squaredSlackNonIncidenceConstraintLinearization(
  y: ReadonlyArray<number>,
  vertexCount: number,
  pair: VertexFaceIncidence,
  d: number
): LightSquaredSlackLinearization {
  const base = incidenceConstraintLinearization(y, vertexCount, pair);
  return {
    value: base.value + d * d,
    gV: base.gV,
    gN: base.gN,
    gB: -1,
    gD: 2 * d,
  };
}
