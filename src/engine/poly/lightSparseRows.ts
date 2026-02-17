import type { Vec3 } from "../math/types";
import { lightBIndex, lightNBase } from "./light";
import {
  incidenceConstraintLinearization,
  squaredSlackNonIncidenceConstraintLinearization,
  unitNormalConstraintLinearization,
} from "./lightConstraints";
import type { VertexFaceIncidence } from "./types";

export type SparseRow = {
  idx: number[];
  val: number[];
  c: number;
};

type IncidenceGradient = {
  gV: Vec3;
  gN: Vec3;
  gB: number;
};

export function pushSparseTriplet(row: SparseRow, idx: number, value: number) {
  if (!Number.isFinite(value)) return;
  if (Math.abs(value) < 1e-14) return;
  row.idx.push(idx);
  row.val.push(value);
}

export function rowsApplyJ(rows: ReadonlyArray<SparseRow>, v: ReadonlyArray<number>): number[] {
  const out = new Array<number>(rows.length).fill(0);
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    let s = 0;
    for (let j = 0; j < row.idx.length; j++) s += row.val[j] * v[row.idx[j]];
    out[i] = s;
  }
  return out;
}

export function rowsApplyJT(rows: ReadonlyArray<SparseRow>, w: ReadonlyArray<number>, dim: number): number[] {
  const out = new Array<number>(dim).fill(0);
  for (let i = 0; i < rows.length; i++) {
    const wi = w[i];
    if (wi === 0) continue;
    const row = rows[i];
    for (let j = 0; j < row.idx.length; j++) out[row.idx[j]] += wi * row.val[j];
  }
  return out;
}

export function pushIncidenceGradientTriplets(
  row: SparseRow,
  vertexCount: number,
  pair: VertexFaceIncidence,
  grad: IncidenceGradient
) {
  const vb = 3 * pair.vi;
  const nb = lightNBase(vertexCount, pair.fi);
  const bb = lightBIndex(vertexCount, pair.fi);
  pushSparseTriplet(row, vb, grad.gV[0]);
  pushSparseTriplet(row, vb + 1, grad.gV[1]);
  pushSparseTriplet(row, vb + 2, grad.gV[2]);
  pushSparseTriplet(row, nb, grad.gN[0]);
  pushSparseTriplet(row, nb + 1, grad.gN[1]);
  pushSparseTriplet(row, nb + 2, grad.gN[2]);
  pushSparseTriplet(row, bb, grad.gB);
}

export function buildIncidenceSparseRow(
  y: ReadonlyArray<number>,
  vertexCount: number,
  pair: VertexFaceIncidence
): SparseRow {
  const lin = incidenceConstraintLinearization(y, vertexCount, pair);
  const row: SparseRow = { idx: [], val: [], c: lin.value };
  pushIncidenceGradientTriplets(row, vertexCount, pair, lin);
  return row;
}

export function buildUnitNormalSparseRow(
  y: ReadonlyArray<number>,
  vertexCount: number,
  fi: number
): SparseRow {
  const lin = unitNormalConstraintLinearization(y, vertexCount, fi);
  const row: SparseRow = { idx: [], val: [], c: lin.value };
  const nb = lightNBase(vertexCount, fi);
  pushSparseTriplet(row, nb, lin.gN[0]);
  pushSparseTriplet(row, nb + 1, lin.gN[1]);
  pushSparseTriplet(row, nb + 2, lin.gN[2]);
  return row;
}

export function buildSquaredSlackNonIncidenceSparseRow(
  y: ReadonlyArray<number>,
  vertexCount: number,
  pair: VertexFaceIncidence,
  d: number,
  dIndex: number
): SparseRow {
  const lin = squaredSlackNonIncidenceConstraintLinearization(y, vertexCount, pair, d);
  const row: SparseRow = { idx: [], val: [], c: lin.value };
  pushIncidenceGradientTriplets(row, vertexCount, pair, lin);
  pushSparseTriplet(row, dIndex, lin.gD);
  return row;
}
