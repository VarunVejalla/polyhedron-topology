import type { Vec3 } from "../math/types";
import {
  buildPolyAuxState,
  buildPolyLightModelFromState,
  buildIncidenceSparseRow,
  buildUnitNormalSparseRow,
  lightFullDim,
  lightNBase,
  lightVertexDim,
  incidenceConstraintLinearization,
  incidenceConstraintValue,
  nonIncidenceConstraintValue,
  pushIncidenceGradientTriplets,
  pushSparseTriplet,
  readLightVertex,
  rowsApplyJ,
  rowsApplyJT,
  unitNormalConstraintValue,
  type SparseRow,
  type PolyState,
  type PolyTopologyData,
  unpackPolyLightState,
} from "../poly";
import { LinearizedAlmEngine } from "../projection/modular/engines/linearizedAlmEngine";
import {
  createArmijoGlobalizer,
  createResidualBalancePenaltyPolicy,
  scaledDualUpdater,
} from "../projection/modular/policies";
import { runMetaSolver } from "../projection/modular/solver";
import type { MetaModel, MetaModelBuilder, MetaState } from "../projection/modular/types";
import { normN } from "../projection/shared/numeric";

type ActivePieceProvider = {
  getTargetAntiFaces: () => number[];
  getActiveEdge: (fi: number) => number | undefined;
};

type FeasibilityOptimizeParams = {
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

type FeasibilityOptimizeDiagnostics = {
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

function buildStateFromLightY(
  y: ReadonlyArray<number>,
  faces: ReadonlyArray<ReadonlyArray<number>>,
  vertexCount: number
): PolyState {
  return unpackPolyLightState(y, faces, vertexCount);
}

function getUnitPlane(
  y: ReadonlyArray<number>,
  vertexCount: number,
  fi: number
): { n: Vec3; b: number } {
  const nb = lightNBase(vertexCount, fi);
  const nx = y[nb];
  const ny = y[nb + 1];
  const nz = y[nb + 2];
  const len = Math.max(1e-12, Math.hypot(nx, ny, nz));
  const inv = 1 / len;
  return {
    n: [nx * inv, ny * inv, nz * inv],
    b: y[nb + 3] * inv,
  };
}

function edgeMargin(
  fi: number,
  edgeIdx: number,
  y: ReadonlyArray<number>,
  faces: ReadonlyArray<ReadonlyArray<number>>,
  vertexCount: number,
  auxState: ReturnType<typeof buildPolyAuxState>
): number {
  const face = faces[fi];
  if (face.length < 3) return Number.POSITIVE_INFINITY;
  const n = getUnitPlane(y, vertexCount, fi).n;
  const q = auxState.projectedComByFace[fi];
  const faceC = auxState.faceCentroid[fi];

  let orientSign = 1;
  for (let i = 0; i < face.length; i++) {
    const a = readLightVertex(y, face[i]);
    const b = readLightVertex(y, face[(i + 1) % face.length]);
    const e = sub3(b, a);
    const s = dot3(cross3(e, sub3(faceC, a)), n);
    if (Math.abs(s) > 1e-12) {
      orientSign = s >= 0 ? 1 : -1;
      break;
    }
  }

  const a = readLightVertex(y, face[edgeIdx]);
  const b = readLightVertex(y, face[(edgeIdx + 1) % face.length]);
  const e = sub3(b, a);
  const len = Math.max(1e-12, norm3(e));
  const s = dot3(cross3(e, sub3(q, a)), n) / len;
  return s * orientSign;
}

function antiResidual(
  fi: number,
  edgeIdx: number,
  y: ReadonlyArray<number>,
  faces: ReadonlyArray<ReadonlyArray<number>>,
  vertexCount: number,
  auxState: ReturnType<typeof buildPolyAuxState>,
  antiMargin: number
): number {
  return edgeMargin(fi, edgeIdx, y, faces, vertexCount, auxState) + antiMargin;
}

class FeasibilityMetaModelBuilder implements MetaModelBuilder {
  private faces: number[][];
  private topology: PolyTopologyData;
  private baselineVertices: Vec3[];
  private paramsProvider: () => FeasibilityOptimizeParams;
  private activePieces: ActivePieceProvider;

  constructor(
    faces: number[][],
    topology: PolyTopologyData,
    baselineVertices: Vec3[],
    paramsProvider: () => FeasibilityOptimizeParams,
    activePieces: ActivePieceProvider
  ) {
    this.faces = faces.map((f) => [...f]);
    this.topology = topology;
    this.baselineVertices = baselineVertices.map((p) => [p[0], p[1], p[2]]);
    this.paramsProvider = paramsProvider;
    this.activePieces = activePieces;
  }

  private vertexCount(): number {
    return this.baselineVertices.length;
  }

  private vertexDim(): number {
    return lightVertexDim(this.vertexCount());
  }

  private fullDim(): number {
    return lightFullDim(this.vertexCount(), this.faces.length);
  }

  private buildAux(y: ReadonlyArray<number>) {
    return buildPolyAuxState(
      buildStateFromLightY(y, this.faces, this.vertexCount()),
      this.topology
    );
  }

  private buildEqRows(y: ReadonlyArray<number>, params: FeasibilityOptimizeParams): SparseRow[] {
    const rows: SparseRow[] = [];

    for (let i = 0; i < this.topology.incidencePairs.length; i++) {
      rows.push(buildIncidenceSparseRow(y, this.vertexCount(), this.topology.incidencePairs[i]));
    }

    for (let fi = 0; fi < this.faces.length; fi++) {
      rows.push(buildUnitNormalSparseRow(y, this.vertexCount(), fi));
    }

    const aux = this.buildAux(y);
    const cVol = aux.volume - params.volumeTarget;
    const rowVol: SparseRow = { idx: [], val: [], c: cVol };
    const eps = 1e-6;
    const yWork = y.slice();
    for (let k = 0; k < this.vertexDim(); k++) {
      const old = yWork[k];
      yWork[k] = old + eps;
      const vp = this.buildAux(yWork).volume;
      yWork[k] = old - eps;
      const vm = this.buildAux(yWork).volume;
      yWork[k] = old;
      pushSparseTriplet(rowVol, k, (vp - vm) / (2 * eps));
    }
    rows.push(rowVol);

    return rows;
  }

  private buildIneqRows(y: ReadonlyArray<number>, params: FeasibilityOptimizeParams): SparseRow[] {
    const rows: SparseRow[] = [];
    const aux = this.buildAux(y);

    for (let i = 0; i < this.topology.nonIncidencePairs.length; i++) {
      const pair = this.topology.nonIncidencePairs[i];
      const g = nonIncidenceConstraintValue(y, this.vertexCount(), pair, params.convexityMargin);
      const row: SparseRow = { idx: [], val: [], c: Math.max(0, g) };
      if (g > 0) {
        const lin = incidenceConstraintLinearization(y, this.vertexCount(), pair);
        pushIncidenceGradientTriplets(row, this.vertexCount(), pair, { ...lin, gB: -1 });
      }
      rows.push(row);
    }

    const eps = 1e-6;
    const yWork = y.slice();
    const targetFaces = this.activePieces.getTargetAntiFaces();
    for (let i = 0; i < targetFaces.length; i++) {
      const fi = targetFaces[i];
      const edgeIdx = this.activePieces.getActiveEdge(fi);
      const row: SparseRow = { idx: [], val: [], c: 0 };
      if (edgeIdx === undefined) {
        rows.push(row);
        continue;
      }
      const g = antiResidual(fi, edgeIdx, y, this.faces, this.vertexCount(), aux, params.antiMargin);
      row.c = Math.max(0, g);
      if (g > 0) {
        for (let k = 0; k < this.vertexDim(); k++) {
          const old = yWork[k];
          yWork[k] = old + eps;
          const gp = antiResidual(fi, edgeIdx, yWork, this.faces, this.vertexCount(), this.buildAux(yWork), params.antiMargin);
          yWork[k] = old - eps;
          const gm = antiResidual(fi, edgeIdx, yWork, this.faces, this.vertexCount(), this.buildAux(yWork), params.antiMargin);
          yWork[k] = old;
          pushSparseTriplet(row, k, (gp - gm) / (2 * eps));
        }
      }
      rows.push(row);
    }

    return rows;
  }

  build(state: Readonly<MetaState>): MetaModel {
    const y = state.y;
    const params = this.paramsProvider();
    const dim = this.fullDim();
    const eqRows = this.buildEqRows(y, params);
    const ineqRows = this.buildIneqRows(y, params);
    const rows = [...eqRows, ...ineqRows];
    const eqCount = eqRows.length;

    const gradient = new Array<number>(dim).fill(0);
    const moveW = Math.max(0, params.moveWeight);
    for (let i = 0; i < this.baselineVertices.length; i++) {
      const b = 3 * i;
      const dx = y[b] - this.baselineVertices[i][0];
      const dy = y[b + 1] - this.baselineVertices[i][1];
      const dz = y[b + 2] - this.baselineVertices[i][2];
      gradient[b] = 2 * moveW * dx;
      gradient[b + 1] = 2 * moveW * dy;
      gradient[b + 2] = 2 * moveW * dz;
    }

    const hDiag = new Array<number>(dim).fill(Math.max(1e-10, params.tau));
    const moveDiag = Math.max(params.tau, 2 * moveW + params.tau);
    for (let i = 0; i < this.vertexDim(); i++) hDiag[i] = moveDiag;

    const evaluateRows = (yy: ReadonlyArray<number>): number[] => {
      const eq = this.buildEqRows(yy, params).map((r) => r.c);
      const ineq = this.buildIneqRows(yy, params).map((r) => r.c);
      return [...eq, ...ineq];
    };

    const c0 = rows.map((r) => r.c);
    return {
      dim,
      gradient,
      hDiag,
      hard: {
        linearization: {
          c0,
          applyJ: (v) => rowsApplyJ(rows, v),
          applyJT: (w) => rowsApplyJT(rows, w, dim),
        },
        evaluate: evaluateRows,
      },
      merit: (yy: ReadonlyArray<number>, u: ReadonlyArray<number>, rho: number) => {
        let f = 0;
        for (let i = 0; i < this.baselineVertices.length; i++) {
          const b = 3 * i;
          const dx = yy[b] - this.baselineVertices[i][0];
          const dy = yy[b + 1] - this.baselineVertices[i][1];
          const dz = yy[b + 2] - this.baselineVertices[i][2];
          f += moveW * (dx * dx + dy * dy + dz * dz);
        }
        const c = evaluateRows(yy);
        let penEq = 0;
        let penIneq = 0;
        for (let i = 0; i < c.length; i++) {
          if (i < eqCount) {
            const t = c[i] + u[i];
            penEq += t * t;
          } else {
            penIneq += c[i] * c[i];
          }
        }
        return f + 0.5 * rho * (penEq + penIneq);
      },
    };
  }
}

class FeasibilityOptimizerSession {
  private faces: number[][];
  private topology: PolyTopologyData;
  private params: FeasibilityOptimizeParams;
  private baselineVertices: Vec3[] = [];
  private iter = 0;

  private state: MetaState = { y: [], u: [], rho: 1 };
  private engine: LinearizedAlmEngine;
  private builder: FeasibilityMetaModelBuilder;

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
    const light = buildPolyLightModelFromState(state);
    this.faces = light.state.faces.map((f) => [...f]);
    this.topology = light.topology;
    this.params = { ...defaultParams, ...params };
    this.baselineVertices = light.state.vertices.map((p) => [p[0], p[1], p[2]]);
    this.stableFaceIndex = this.resolveStableFaceIndex(this.params.stableFaceIndex);
    this.targetAntiFaces = this.faces.map((_f, fi) => fi).filter((fi) => fi !== this.stableFaceIndex);

    this.state = {
      y: new Array<number>(lightFullDim(this.baselineVertices.length, this.faces.length)).fill(0),
      u: [],
      rho: Math.max(this.params.rhoMin, Math.min(this.params.rhoMax, this.params.rho)),
    };
    for (let i = 0; i < this.baselineVertices.length; i++) {
      const b = 3 * i;
      this.state.y[b] = light.state.vertices[i][0];
      this.state.y[b + 1] = light.state.vertices[i][1];
      this.state.y[b + 2] = light.state.vertices[i][2];
    }
    for (let fi = 0; fi < this.faces.length; fi++) {
      const nb = lightNBase(this.baselineVertices.length, fi);
      this.state.y[nb] = light.state.facePlanes[fi].n[0];
      this.state.y[nb + 1] = light.state.facePlanes[fi].n[1];
      this.state.y[nb + 2] = light.state.facePlanes[fi].n[2];
      this.state.y[nb + 3] = light.state.facePlanes[fi].b;
    }

    this.engine = new LinearizedAlmEngine({
      cgIters: Math.max(8, this.params.cgIters),
      cgTol: Math.max(1e-10, this.params.cgTol),
    });
    this.builder = new FeasibilityMetaModelBuilder(
      this.faces,
      this.topology,
      this.baselineVertices,
      () => this.params,
      {
        getTargetAntiFaces: () => this.targetAntiFaces,
        getActiveEdge: (fi: number) => this.activeEdgeByFace.get(fi),
      }
    );

    this.refreshAntiPieces(true);
    const eqCount = this.eqConstraintCount();
    const ineqCount = this.ineqConstraintCount();
    this.state.u = new Array(eqCount + ineqCount).fill(0);
    this.lastDiag = this.computeDiagnostics(this.state.y);
  }

  private vertexCount(): number {
    return this.baselineVertices.length;
  }

  private resolveStableFaceIndex(candidate: number): number {
    if (this.faces.length === 0) return -1;
    const idx = Math.floor(candidate);
    if (!Number.isFinite(idx)) return 0;
    if (idx < 0) return 0;
    if (idx >= this.faces.length) return this.faces.length - 1;
    return idx;
  }

  private buildAux(y: ReadonlyArray<number>) {
    return buildPolyAuxState(buildStateFromLightY(y, this.faces, this.vertexCount()), this.topology);
  }

  private minFaceMargin(fi: number, y: ReadonlyArray<number>, aux: ReturnType<typeof buildPolyAuxState>): { edge: number; margin: number } {
    const face = this.faces[fi];
    if (face.length < 3) return { edge: -1, margin: Number.POSITIVE_INFINITY };
    let bestEdge = 0;
    let bestMargin = Number.POSITIVE_INFINITY;
    for (let ei = 0; ei < face.length; ei++) {
      const m = edgeMargin(fi, ei, y, this.faces, this.vertexCount(), aux);
      if (m < bestMargin) {
        bestMargin = m;
        bestEdge = ei;
      }
    }
    return { edge: bestEdge, margin: bestMargin };
  }

  private refreshAntiPieces(force: boolean) {
    const aux = this.buildAux(this.state.y);
    for (const fi of this.targetAntiFaces) {
      const candidate = this.minFaceMargin(fi, this.state.y, aux);
      const curEdge = this.activeEdgeByFace.get(fi);
      const curDwell = this.dwellByFace.get(fi) ?? 0;
      if (force || curEdge === undefined) {
        this.activeEdgeByFace.set(fi, candidate.edge);
        this.dwellByFace.set(fi, 0);
        continue;
      }
      const curMargin = edgeMargin(fi, curEdge, this.state.y, this.faces, this.vertexCount(), aux);
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

  private eqConstraintCount(): number {
    return this.topology.incidencePairs.length + this.faces.length + 1;
  }

  private ineqConstraintCount(): number {
    return this.topology.nonIncidencePairs.length + this.targetAntiFaces.length;
  }

  private computeEqResiduals(y: ReadonlyArray<number>): number[] {
    const out: number[] = [];
    for (let i = 0; i < this.topology.incidencePairs.length; i++) {
      out.push(incidenceConstraintValue(y, this.vertexCount(), this.topology.incidencePairs[i]));
    }
    for (let fi = 0; fi < this.faces.length; fi++) {
      out.push(unitNormalConstraintValue(y, this.vertexCount(), fi));
    }
    out.push(this.buildAux(y).volume - this.params.volumeTarget);
    return out;
  }

  private computeIneqDiagnostics(y: ReadonlyArray<number>): {
    maxViolation: number;
    activeConvexityCount: number;
    activeAntiCount: number;
  } {
    const aux = this.buildAux(y);
    let maxViolation = 0;
    let activeConvexityCount = 0;
    let activeAntiCount = 0;

    for (let i = 0; i < this.topology.nonIncidencePairs.length; i++) {
      const pair = this.topology.nonIncidencePairs[i];
      const g = nonIncidenceConstraintValue(y, this.vertexCount(), pair, this.params.convexityMargin);
      if (g > 0) {
        activeConvexityCount++;
        if (g > maxViolation) maxViolation = g;
      }
    }

    for (let i = 0; i < this.targetAntiFaces.length; i++) {
      const fi = this.targetAntiFaces[i];
      const edgeIdx = this.activeEdgeByFace.get(fi);
      if (edgeIdx === undefined) continue;
      const g = antiResidual(fi, edgeIdx, y, this.faces, this.vertexCount(), aux, this.params.antiMargin);
      if (g > 0) {
        activeAntiCount++;
        if (g > maxViolation) maxViolation = g;
      }
    }

    return { maxViolation, activeConvexityCount, activeAntiCount };
  }

  private computeDiagnostics(y: ReadonlyArray<number>): FeasibilityOptimizeDiagnostics {
    const eq = this.computeEqResiduals(y);
    const ineq = this.computeIneqDiagnostics(y);
    const volume = this.buildAux(y).volume;
    return {
      iter: this.iter,
      eqResidualL2: normN(eq),
      ineqViolationMax: ineq.maxViolation,
      volume,
      activeConvexityCount: ineq.activeConvexityCount,
      activeAntiCount: ineq.activeAntiCount,
    };
  }

  private shouldStop(diag: FeasibilityOptimizeDiagnostics): boolean {
    return diag.eqResidualL2 <= this.params.tolEq && diag.ineqViolationMax <= this.params.tolIneq;
  }

  step(maxIters: number): boolean {
    const budget = Math.max(1, Math.floor(maxIters));
    for (let k = 0; k < budget; k++) {
      if (this.iter >= this.params.maxOuterIters) return true;
      this.iter++;
      this.refreshAntiPieces(false);

      this.engine.setParams({
        cgIters: Math.max(8, this.params.cgIters),
        cgTol: Math.max(1e-10, this.params.cgTol),
      });

      const globalizer = createArmijoGlobalizer({
        c1: 1e-4,
        shrink: Math.min(0.95, Math.max(0.1, this.params.lineSearchShrink)),
        maxSteps: Math.max(1, this.params.lineSearchMaxSteps),
      });
      const penaltyPolicy = createResidualBalancePenaltyPolicy({
        enabled: this.params.adaptRho,
        increase: Math.max(1.01, this.params.rhoIncrease),
        decrease: Math.max(1.01, this.params.rhoDecrease),
        ratio: Math.max(1.1, this.params.rhoResidualRatio),
        min: Math.max(1e-8, this.params.rhoMin),
        max: Math.max(Math.max(1e-8, this.params.rhoMin), this.params.rhoMax),
      });

      const eqCount = this.eqConstraintCount();
      const dualUpdate = {
        update: (state: MetaState, cNew: ReadonlyArray<number>) => {
          scaledDualUpdater.update(state, cNew);
          for (let i = eqCount; i < state.u.length; i++) state.u[i] = 0;
        },
      };

      const stats = runMetaSolver(
        this.state,
        this.builder,
        this.engine,
        globalizer,
        dualUpdate,
        penaltyPolicy,
        undefined,
        1
      );

      if (stats.accepted === 0 || stats.lastAlpha < this.params.minAcceptedAlpha) {
        this.lastDiag = this.computeDiagnostics(this.state.y);
        return this.shouldStop(this.lastDiag) || this.iter >= this.params.maxOuterIters;
      }

      this.lastDiag = this.computeDiagnostics(this.state.y);
      if (this.shouldStop(this.lastDiag)) return true;
    }
    return this.iter >= this.params.maxOuterIters || this.shouldStop(this.lastDiag);
  }

  getVertices(): Vec3[] {
    const out: Vec3[] = new Array(this.vertexCount());
    for (let i = 0; i < this.vertexCount(); i++) out[i] = readLightVertex(this.state.y, i);
    return out;
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
