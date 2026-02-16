import type { Vec3 } from "../math/types";
import type { PolyState } from "../poly";
import { normN, solveCG } from "../projection/shared/numeric";

type SparseRow = {
  idx: number[];
  val: number[];
  c: number;
};

export type FeasibilityOptimizeParams = {
  rho: number;
  tau: number;
  maxOuterIters: number;
  cgIters: number;
  cgTol: number;
  lineSearchShrink: number;
  lineSearchMaxSteps: number;
  minAcceptedAlpha: number;
  tolEq: number;
  tolIneq: number;
  moveWeight: number;
  volumeTarget: number;
  convexityMargin: number;
  antiMargin: number;
  antiSwitchTol: number;
  antiMinDwell: number;
  adaptRho: boolean;
  rhoIncrease: number;
  rhoDecrease: number;
  rhoResidualRatio: number;
  rhoMin: number;
  rhoMax: number;
  stableFaceIndex: number;
};

export type FeasibilityOptimizeDiagnostics = {
  iter: number;
  eqResidualL2: number;
  ineqViolationMax: number;
  volume: number;
  activeConvexityCount: number;
  activeAntiCount: number;
};

const defaultParams: FeasibilityOptimizeParams = {
  rho: 25,
  tau: 1e-6,
  maxOuterIters: 80,
  cgIters: 120,
  cgTol: 1e-6,
  lineSearchShrink: 0.5,
  lineSearchMaxSteps: 10,
  minAcceptedAlpha: 1e-4,
  tolEq: 1e-5,
  tolIneq: 1e-5,
  moveWeight: 1e-3,
  volumeTarget: 1,
  convexityMargin: 0,
  antiMargin: 0.01,
  antiSwitchTol: 1e-3,
  antiMinDwell: 3,
  adaptRho: true,
  rhoIncrease: 2,
  rhoDecrease: 2,
  rhoResidualRatio: 10,
  rhoMin: 1e-3,
  rhoMax: 1e7,
  stableFaceIndex: 0,
};

function dot3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function sub3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function add3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function mul3(a: ReadonlyArray<number>, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function cross3(a: ReadonlyArray<number>, b: ReadonlyArray<number>): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function norm3(a: ReadonlyArray<number>): number {
  return Math.hypot(a[0], a[1], a[2]);
}

function pushSparseTriplet(row: SparseRow, idx: number, value: number) {
  if (!Number.isFinite(value)) return;
  if (Math.abs(value) < 1e-14) return;
  row.idx.push(idx);
  row.val.push(value);
}

export class FeasibilityOptimizerSession {
  private faces: number[][];
  private incidencePairs: Array<{ fi: number; vi: number }> = [];
  private nonIncidencePairs: Array<{ fi: number; vi: number }> = [];
  private params: FeasibilityOptimizeParams;

  private y: number[] = [];
  private baselineVertices: Vec3[] = [];
  private lambdaEq: number[] = [];
  private rho: number;
  private iter = 0;

  private targetAntiFaces: number[] = [];
  private stableFaceIndex = 0;
  private activeEdgeByFace = new Map<number, number>();
  private dwellByFace = new Map<number, number>();

  private lastDiag: FeasibilityOptimizeDiagnostics = {
    iter: 0,
    eqResidualL2: Number.POSITIVE_INFINITY,
    ineqViolationMax: Number.POSITIVE_INFINITY,
    volume: 0,
    activeConvexityCount: 0,
    activeAntiCount: 0,
  };

  constructor(state: PolyState, params?: Partial<FeasibilityOptimizeParams>) {
    this.faces = state.faces.map((f) => [...f]);
    this.params = { ...defaultParams, ...params };
    this.rho = Math.max(this.params.rhoMin, Math.min(this.params.rhoMax, this.params.rho));
    this.baselineVertices = state.vertices.map((p) => [p[0], p[1], p[2]] as Vec3);
    this.buildIncidence();
    this.stableFaceIndex = this.resolveStableFaceIndex(this.params.stableFaceIndex);
    this.y = this.packInitialState(state);
    this.refreshAntiPieces(true);
    const eq = this.evalEqResiduals(this.y);
    this.lambdaEq = new Array(eq.length).fill(0);
    this.lastDiag = this.computeDiagnostics(this.y);
  }

  private vertexCount(): number {
    return this.baselineVertices.length;
  }

  private faceCount(): number {
    return this.faces.length;
  }

  private vertexDim(): number {
    return 3 * this.vertexCount();
  }

  private fullDim(): number {
    return this.vertexDim() + 4 * this.faceCount();
  }

  private nBase(fi: number): number {
    return this.vertexDim() + 4 * fi;
  }

  private packInitialState(state: PolyState): number[] {
    const dim = this.fullDim();
    const y = new Array<number>(dim).fill(0);
    for (let i = 0; i < state.vertices.length; i++) {
      const b = 3 * i;
      y[b] = state.vertices[i][0];
      y[b + 1] = state.vertices[i][1];
      y[b + 2] = state.vertices[i][2];
    }
    for (let fi = 0; fi < this.faces.length; fi++) {
      const nb = this.nBase(fi);
      const pl = state.facePlanes[fi];
      y[nb] = pl.n[0];
      y[nb + 1] = pl.n[1];
      y[nb + 2] = pl.n[2];
      y[nb + 3] = pl.b;
    }
    return y;
  }

  private buildIncidence() {
    this.incidencePairs = [];
    const isInc = new Array<Set<number>>(this.faces.length);
    for (let fi = 0; fi < this.faces.length; fi++) {
      isInc[fi] = new Set<number>(this.faces[fi]);
      for (const vi of this.faces[fi]) this.incidencePairs.push({ fi, vi });
    }

    this.nonIncidencePairs = [];
    for (let fi = 0; fi < this.faces.length; fi++) {
      const set = isInc[fi];
      for (let vi = 0; vi < this.vertexCount(); vi++) {
        if (!set.has(vi)) this.nonIncidencePairs.push({ fi, vi });
      }
    }
  }

  private unpackVertices(y: ReadonlyArray<number>): Vec3[] {
    const out: Vec3[] = new Array(this.vertexCount());
    for (let i = 0; i < this.vertexCount(); i++) {
      const b = 3 * i;
      out[i] = [y[b], y[b + 1], y[b + 2]];
    }
    return out;
  }

  private getUnitPlane(fi: number, y: ReadonlyArray<number>): { n: Vec3; b: number } {
    const nb = this.nBase(fi);
    const nx = y[nb];
    const ny = y[nb + 1];
    const nz = y[nb + 2];
    const nLen = Math.max(1e-12, Math.hypot(nx, ny, nz));
    const inv = 1 / nLen;
    const n: Vec3 = [nx * inv, ny * inv, nz * inv];
    const b = y[nb + 3] * inv;
    return { n, b };
  }

  private computeVolumeAndCom(y: ReadonlyArray<number>): { volume: number; centerOfMass: Vec3 } {
    const v = this.unpackVertices(y);
    const ref: Vec3 = [0, 0, 0];
    for (let i = 0; i < v.length; i++) {
      ref[0] += v[i][0];
      ref[1] += v[i][1];
      ref[2] += v[i][2];
    }
    if (v.length > 0) {
      const inv = 1 / v.length;
      ref[0] *= inv;
      ref[1] *= inv;
      ref[2] *= inv;
    }

    let totalVol = 0;
    let comNum: Vec3 = [0, 0, 0];
    for (let fi = 0; fi < this.faces.length; fi++) {
      const face = this.faces[fi];
      if (face.length < 3) continue;
      const plane = this.getUnitPlane(fi, y);
      const a0 = v[face[0]];
      for (let i = 1; i + 1 < face.length; i++) {
        let b = v[face[i]];
        let c = v[face[i + 1]];
        let triN = cross3(sub3(b, a0), sub3(c, a0));
        if (dot3(triN, plane.n) < 0) {
          const tmp = b;
          b = c;
          c = tmp;
          triN = cross3(sub3(b, a0), sub3(c, a0));
        }

        const pa = sub3(a0, ref);
        const pb = sub3(b, ref);
        const pc = sub3(c, ref);
        let vol = dot3(pa, cross3(pb, pc)) / 6;
        if (vol < 0) vol = -vol;
        if (vol <= 1e-15) continue;

        const tetraC = mul3(add3(add3(add3(ref, a0), b), c), 0.25);
        comNum[0] += tetraC[0] * vol;
        comNum[1] += tetraC[1] * vol;
        comNum[2] += tetraC[2] * vol;
        totalVol += vol;
      }
    }

    if (totalVol <= 1e-15) return { volume: 0, centerOfMass: ref };
    return {
      volume: totalVol,
      centerOfMass: [comNum[0] / totalVol, comNum[1] / totalVol, comNum[2] / totalVol],
    };
  }

  private faceCentroid(fi: number, y: ReadonlyArray<number>): Vec3 {
    const face = this.faces[fi];
    const c: Vec3 = [0, 0, 0];
    if (face.length === 0) return c;
    for (let i = 0; i < face.length; i++) {
      const vb = 3 * face[i];
      c[0] += y[vb];
      c[1] += y[vb + 1];
      c[2] += y[vb + 2];
    }
    const inv = 1 / face.length;
    c[0] *= inv;
    c[1] *= inv;
    c[2] *= inv;
    return c;
  }

  private edgeMargin(fi: number, edgeIdx: number, y: ReadonlyArray<number>): number {
    const face = this.faces[fi];
    if (face.length < 3) return Number.POSITIVE_INFINITY;
    const { n, b } = this.getUnitPlane(fi, y);
    const { centerOfMass } = this.computeVolumeAndCom(y);
    const d = dot3(n, centerOfMass) - b;
    const q: Vec3 = [centerOfMass[0] - n[0] * d, centerOfMass[1] - n[1] * d, centerOfMass[2] - n[2] * d];

    const faceC = this.faceCentroid(fi, y);
    let orientSign = 1;
    for (let i = 0; i < face.length; i++) {
      const aIdx = face[i];
      const bIdx = face[(i + 1) % face.length];
      const a: Vec3 = [y[3 * aIdx], y[3 * aIdx + 1], y[3 * aIdx + 2]];
      const bb: Vec3 = [y[3 * bIdx], y[3 * bIdx + 1], y[3 * bIdx + 2]];
      const e = sub3(bb, a);
      const s = dot3(cross3(e, sub3(faceC, a)), n);
      if (Math.abs(s) > 1e-12) {
        orientSign = s >= 0 ? 1 : -1;
        break;
      }
    }

    const aIdx = face[edgeIdx];
    const bIdx = face[(edgeIdx + 1) % face.length];
    const a: Vec3 = [y[3 * aIdx], y[3 * aIdx + 1], y[3 * aIdx + 2]];
    const bb: Vec3 = [y[3 * bIdx], y[3 * bIdx + 1], y[3 * bIdx + 2]];
    const e = sub3(bb, a);
    const len = Math.max(1e-12, norm3(e));
    const s = dot3(cross3(e, sub3(q, a)), n) / len;
    return s * orientSign;
  }

  private minFaceMargin(fi: number, y: ReadonlyArray<number>): { edge: number; margin: number } {
    const face = this.faces[fi];
    if (face.length < 3) return { edge: -1, margin: Number.POSITIVE_INFINITY };
    let bestEdge = 0;
    let bestMargin = Number.POSITIVE_INFINITY;
    for (let ei = 0; ei < face.length; ei++) {
      const m = this.edgeMargin(fi, ei, y);
      if (m < bestMargin) {
        bestMargin = m;
        bestEdge = ei;
      }
    }
    return { edge: bestEdge, margin: bestMargin };
  }

  private resolveStableFaceIndex(candidate: number): number {
    if (this.faces.length === 0) return -1;
    const idx = Math.floor(candidate);
    if (!Number.isFinite(idx)) return 0;
    if (idx < 0) return 0;
    if (idx >= this.faces.length) return this.faces.length - 1;
    return idx;
  }

  private refreshAntiPieces(force: boolean) {
    const targets: number[] = [];
    for (let fi = 0; fi < this.faces.length; fi++) {
      if (fi === this.stableFaceIndex) continue;
      targets.push(fi);
    }
    this.targetAntiFaces = targets;

    for (const fi of this.targetAntiFaces) {
      const candidate = this.minFaceMargin(fi, this.y);
      const curEdge = this.activeEdgeByFace.get(fi);
      const curDwell = this.dwellByFace.get(fi) ?? 0;
      if (force || curEdge === undefined) {
        this.activeEdgeByFace.set(fi, candidate.edge);
        this.dwellByFace.set(fi, 0);
        continue;
      }
      const curMargin = this.edgeMargin(fi, curEdge, this.y);
      const shouldSwitch =
        candidate.edge !== curEdge &&
        curDwell >= this.params.antiMinDwell &&
        candidate.margin < curMargin - this.params.antiSwitchTol;
      if (shouldSwitch) {
        this.activeEdgeByFace.set(fi, candidate.edge);
        this.dwellByFace.set(fi, 0);
      } else {
        this.dwellByFace.set(fi, curDwell + 1);
      }
    }
  }

  private evalEqResiduals(y: ReadonlyArray<number>): number[] {
    const out: number[] = [];
    for (let i = 0; i < this.incidencePairs.length; i++) {
      const { fi, vi } = this.incidencePairs[i];
      const vb = 3 * vi;
      const nb = this.nBase(fi);
      const c =
        y[nb] * y[vb] +
        y[nb + 1] * y[vb + 1] +
        y[nb + 2] * y[vb + 2] -
        y[nb + 3];
      out.push(c);
    }
    for (let fi = 0; fi < this.faces.length; fi++) {
      const nb = this.nBase(fi);
      const c = y[nb] * y[nb] + y[nb + 1] * y[nb + 1] + y[nb + 2] * y[nb + 2] - 1;
      out.push(c);
    }
    out.push(this.computeVolumeAndCom(y).volume - this.params.volumeTarget);
    return out;
  }

  private buildEqRows(y: ReadonlyArray<number>): { rows: SparseRow[]; cEq: number[] } {
    const rows: SparseRow[] = [];
    const cEq: number[] = [];

    for (let i = 0; i < this.incidencePairs.length; i++) {
      const { fi, vi } = this.incidencePairs[i];
      const vb = 3 * vi;
      const nb = this.nBase(fi);
      const c =
        y[nb] * y[vb] +
        y[nb + 1] * y[vb + 1] +
        y[nb + 2] * y[vb + 2] -
        y[nb + 3];
      const row: SparseRow = { idx: [], val: [], c };
      pushSparseTriplet(row, vb, y[nb]);
      pushSparseTriplet(row, vb + 1, y[nb + 1]);
      pushSparseTriplet(row, vb + 2, y[nb + 2]);
      pushSparseTriplet(row, nb, y[vb]);
      pushSparseTriplet(row, nb + 1, y[vb + 1]);
      pushSparseTriplet(row, nb + 2, y[vb + 2]);
      pushSparseTriplet(row, nb + 3, -1);
      rows.push(row);
      cEq.push(c);
    }

    for (let fi = 0; fi < this.faces.length; fi++) {
      const nb = this.nBase(fi);
      const c = y[nb] * y[nb] + y[nb + 1] * y[nb + 1] + y[nb + 2] * y[nb + 2] - 1;
      const row: SparseRow = { idx: [], val: [], c };
      pushSparseTriplet(row, nb, 2 * y[nb]);
      pushSparseTriplet(row, nb + 1, 2 * y[nb + 1]);
      pushSparseTriplet(row, nb + 2, 2 * y[nb + 2]);
      rows.push(row);
      cEq.push(c);
    }

    const vol = this.computeVolumeAndCom(y).volume;
    const cVol = vol - this.params.volumeTarget;
    const rowVol: SparseRow = { idx: [], val: [], c: cVol };
    const eps = 1e-6;
    const yWork = y.slice();
    for (let k = 0; k < this.vertexDim(); k++) {
      const old = yWork[k];
      yWork[k] = old + eps;
      const vp = this.computeVolumeAndCom(yWork).volume;
      yWork[k] = old - eps;
      const vm = this.computeVolumeAndCom(yWork).volume;
      yWork[k] = old;
      const g = (vp - vm) / (2 * eps);
      pushSparseTriplet(rowVol, k, g);
    }
    rows.push(rowVol);
    cEq.push(cVol);

    return { rows, cEq };
  }

  private antiResidual(fi: number, edgeIdx: number, y: ReadonlyArray<number>): number {
    return this.edgeMargin(fi, edgeIdx, y) + this.params.antiMargin;
  }

  private buildIneqRows(y: ReadonlyArray<number>): {
    rows: SparseRow[];
    maxViolation: number;
    activeConvexityCount: number;
    activeAntiCount: number;
  } {
    const rows: SparseRow[] = [];
    let maxViolation = 0;
    let activeConvexityCount = 0;
    let activeAntiCount = 0;

    for (let i = 0; i < this.nonIncidencePairs.length; i++) {
      const { fi, vi } = this.nonIncidencePairs[i];
      const vb = 3 * vi;
      const nb = this.nBase(fi);
      const g =
        y[nb] * y[vb] +
        y[nb + 1] * y[vb + 1] +
        y[nb + 2] * y[vb + 2] -
        y[nb + 3] +
        this.params.convexityMargin;
      if (g <= 0) continue;
      const row: SparseRow = { idx: [], val: [], c: g };
      pushSparseTriplet(row, vb, y[nb]);
      pushSparseTriplet(row, vb + 1, y[nb + 1]);
      pushSparseTriplet(row, vb + 2, y[nb + 2]);
      pushSparseTriplet(row, nb, y[vb]);
      pushSparseTriplet(row, nb + 1, y[vb + 1]);
      pushSparseTriplet(row, nb + 2, y[vb + 2]);
      pushSparseTriplet(row, nb + 3, -1);
      rows.push(row);
      activeConvexityCount++;
      if (g > maxViolation) maxViolation = g;
    }

    const eps = 1e-6;
    const yWork = y.slice();
    for (const fi of this.targetAntiFaces) {
      const edgeIdx = this.activeEdgeByFace.get(fi);
      if (edgeIdx === undefined) continue;
      const g = this.antiResidual(fi, edgeIdx, y);
      if (g <= 0) continue;
      const row: SparseRow = { idx: [], val: [], c: g };
      for (let k = 0; k < this.vertexDim(); k++) {
        const old = yWork[k];
        yWork[k] = old + eps;
        const gp = this.antiResidual(fi, edgeIdx, yWork);
        yWork[k] = old - eps;
        const gm = this.antiResidual(fi, edgeIdx, yWork);
        yWork[k] = old;
        const grad = (gp - gm) / (2 * eps);
        pushSparseTriplet(row, k, grad);
      }
      rows.push(row);
      activeAntiCount++;
      if (g > maxViolation) maxViolation = g;
    }

    return { rows, maxViolation, activeConvexityCount, activeAntiCount };
  }

  private rowDot(row: SparseRow, x: ReadonlyArray<number>): number {
    let s = 0;
    for (let j = 0; j < row.idx.length; j++) s += row.val[j] * x[row.idx[j]];
    return s;
  }

  private jtTimes(rows: ReadonlyArray<SparseRow>, coeffs: ReadonlyArray<number>): number[] {
    const out = new Array<number>(this.fullDim()).fill(0);
    for (let i = 0; i < rows.length; i++) {
      const c = coeffs[i];
      if (c === 0) continue;
      const row = rows[i];
      for (let j = 0; j < row.idx.length; j++) out[row.idx[j]] += c * row.val[j];
    }
    return out;
  }

  private applyA(
    v: ReadonlyArray<number>,
    hDiag: ReadonlyArray<number>,
    eqRows: ReadonlyArray<SparseRow>,
    ineqRows: ReadonlyArray<SparseRow>
  ): number[] {
    const out = new Array<number>(this.fullDim());
    for (let i = 0; i < out.length; i++) out[i] = hDiag[i] * v[i];

    for (let i = 0; i < eqRows.length; i++) {
      const row = eqRows[i];
      const s = this.rowDot(row, v);
      const k = this.rho * s;
      for (let j = 0; j < row.idx.length; j++) out[row.idx[j]] += k * row.val[j];
    }
    for (let i = 0; i < ineqRows.length; i++) {
      const row = ineqRows[i];
      const s = this.rowDot(row, v);
      const k = this.rho * s;
      for (let j = 0; j < row.idx.length; j++) out[row.idx[j]] += k * row.val[j];
    }
    return out;
  }

  private objectiveAndGradient(y: ReadonlyArray<number>, gradOut?: number[]): number {
    const moveW = Math.max(0, this.params.moveWeight);
    if (moveW <= 0) {
      if (gradOut) for (let i = 0; i < gradOut.length; i++) gradOut[i] = 0;
      return 0;
    }

    let f = 0;
    if (gradOut) for (let i = 0; i < gradOut.length; i++) gradOut[i] = 0;
    for (let i = 0; i < this.vertexCount(); i++) {
      const b = 3 * i;
      const dx = y[b] - this.baselineVertices[i][0];
      const dy = y[b + 1] - this.baselineVertices[i][1];
      const dz = y[b + 2] - this.baselineVertices[i][2];
      f += moveW * (dx * dx + dy * dy + dz * dz);
      if (gradOut) {
        gradOut[b] += 2 * moveW * dx;
        gradOut[b + 1] += 2 * moveW * dy;
        gradOut[b + 2] += 2 * moveW * dz;
      }
    }
    return f;
  }

  private merit(y: ReadonlyArray<number>): number {
    const eq = this.evalEqResiduals(y);
    const { rows: ineqRows } = this.buildIneqRows(y);
    const f = this.objectiveAndGradient(y);
    let penEq = 0;
    for (let i = 0; i < eq.length; i++) {
      const t = eq[i] + this.lambdaEq[i] / this.rho;
      penEq += t * t;
    }
    let penIneq = 0;
    for (let i = 0; i < ineqRows.length; i++) penIneq += ineqRows[i].c * ineqRows[i].c;
    return f + 0.5 * this.rho * (penEq + penIneq);
  }

  private computeDiagnostics(y: ReadonlyArray<number>): FeasibilityOptimizeDiagnostics {
    const eq = this.evalEqResiduals(y);
    const ineq = this.buildIneqRows(y);
    const vol = this.computeVolumeAndCom(y).volume;
    return {
      iter: this.iter,
      eqResidualL2: normN(eq),
      ineqViolationMax: ineq.maxViolation,
      volume: vol,
      activeConvexityCount: ineq.activeConvexityCount,
      activeAntiCount: ineq.activeAntiCount,
    };
  }

  private clampRho(next: number): number {
    return Math.max(this.params.rhoMin, Math.min(this.params.rhoMax, next));
  }

  private shouldStop(diag: FeasibilityOptimizeDiagnostics): boolean {
    return diag.eqResidualL2 <= this.params.tolEq && diag.ineqViolationMax <= this.params.tolIneq;
  }

  step(maxIters: number): boolean {
    const budget = Math.max(1, Math.floor(maxIters));
    for (let local = 0; local < budget; local++) {
      if (this.iter >= this.params.maxOuterIters) return true;
      this.iter++;
      this.refreshAntiPieces(false);

      const dim = this.fullDim();
      const grad = new Array<number>(dim).fill(0);
      this.objectiveAndGradient(this.y, grad);

      const hDiag = new Array<number>(dim).fill(this.params.tau);
      const moveDiag = Math.max(this.params.tau, 2 * Math.max(0, this.params.moveWeight) + this.params.tau);
      for (let i = 0; i < this.vertexDim(); i++) hDiag[i] = moveDiag;

      const { rows: eqRows, cEq } = this.buildEqRows(this.y);
      if (this.lambdaEq.length !== eqRows.length) this.lambdaEq = new Array(eqRows.length).fill(0);
      const { rows: ineqRows } = this.buildIneqRows(this.y);

      const eqCoeff = new Array<number>(eqRows.length);
      for (let i = 0; i < eqRows.length; i++) eqCoeff[i] = this.lambdaEq[i] + this.rho * cEq[i];
      const gradEq = this.jtTimes(eqRows, eqCoeff);

      const ineqCoeff = new Array<number>(ineqRows.length);
      for (let i = 0; i < ineqRows.length; i++) ineqCoeff[i] = this.rho * ineqRows[i].c;
      const gradIneq = this.jtTimes(ineqRows, ineqCoeff);

      const rhs = new Array<number>(dim);
      for (let i = 0; i < dim; i++) rhs[i] = grad[i] + gradEq[i] + gradIneq[i];

      const b = rhs.map((r) => -r);
      const delta = solveCG(
        (v) => this.applyA(v, hDiag, eqRows, ineqRows),
        b,
        Math.max(8, this.params.cgIters),
        Math.max(1e-10, this.params.cgTol)
      );

      const stepNorm = normN(delta);
      if (!Number.isFinite(stepNorm) || stepNorm <= 1e-12) {
        this.lastDiag = this.computeDiagnostics(this.y);
        return this.shouldStop(this.lastDiag) || this.iter >= this.params.maxOuterIters;
      }

      const merit0 = this.merit(this.y);
      let alpha = 1;
      let accepted = false;
      let nextY = this.y;
      for (let ls = 0; ls < this.params.lineSearchMaxSteps; ls++) {
        const trial = new Array<number>(dim);
        for (let i = 0; i < dim; i++) trial[i] = this.y[i] + alpha * delta[i];
        const mTrial = this.merit(trial);
        if (mTrial <= merit0) {
          nextY = trial;
          accepted = true;
          break;
        }
        alpha *= this.params.lineSearchShrink;
      }
      if (!accepted || alpha < this.params.minAcceptedAlpha) {
        this.lastDiag = this.computeDiagnostics(this.y);
        return this.shouldStop(this.lastDiag) || this.iter >= this.params.maxOuterIters;
      }

      const oldEq = this.evalEqResiduals(this.y);
      this.y = nextY;
      const newEq = this.evalEqResiduals(this.y);

      for (let i = 0; i < this.lambdaEq.length; i++) this.lambdaEq[i] += this.rho * newEq[i];

      if (this.params.adaptRho) {
        const primal = normN(newEq);
        const dEq = new Array<number>(newEq.length);
        for (let i = 0; i < newEq.length; i++) dEq[i] = newEq[i] - oldEq[i];
        const dual = this.rho * normN(this.jtTimes(eqRows, dEq));

        let rhoNew = this.rho;
        if (primal > this.params.rhoResidualRatio * dual) {
          rhoNew = this.clampRho(this.rho * this.params.rhoIncrease);
        } else if (dual > this.params.rhoResidualRatio * primal) {
          rhoNew = this.clampRho(this.rho / this.params.rhoDecrease);
        }

        if (rhoNew !== this.rho) {
          const scale = this.rho / rhoNew;
          for (let i = 0; i < this.lambdaEq.length; i++) this.lambdaEq[i] *= scale;
          this.rho = rhoNew;
        }
      }

      this.lastDiag = this.computeDiagnostics(this.y);
      if (this.shouldStop(this.lastDiag)) return true;
    }
    return this.iter >= this.params.maxOuterIters || this.shouldStop(this.lastDiag);
  }

  getVertices(): Vec3[] {
    return this.unpackVertices(this.y);
  }

  getDiagnostics(): FeasibilityOptimizeDiagnostics {
    return this.lastDiag;
  }
}

export function createFeasibilityOptimizerSession(
  state: PolyState,
  params?: Partial<FeasibilityOptimizeParams>
): FeasibilityOptimizerSession {
  return new FeasibilityOptimizerSession(state, params);
}
