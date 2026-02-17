import type { Vec3 } from "../math/types";
import {
  buildPolyAuxState,
  buildPolyLightModelFromState,
  incidenceConstraintValue,
  lightFullDim,
  lightNBase,
  lightVertexDim,
  nonIncidenceConstraintValue,
  packPolyLightState,
  readLightVertex,
  type PolyState,
  type PolyTopologyData,
  type VertexFaceIncidence,
} from "../poly";
import {
  linearConstraintAsQuadratic,
  sparseSymmetricOperator,
} from "../optimization/quadratic";
import { LinearizedAlmKernel } from "../optimization/kernels/linearizedAlmKernel";
import type {
  OptimizationProblem,
  OptimizerHyperParams,
  OptimizerState,
  ProblemBuilder,
  QuadraticConstraint,
  SymmetricEntry,
} from "../optimization/types";
import { normN } from "../projection/shared/numeric";

type ActivePieceProvider = {
  getTargetAntiFaces: () => number[];
  getActiveEdge: (fi: number) => number | undefined;
};

type SparseRow = {
  idx: number[];
  val: number[];
  c: number;
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
  const nb = lightNBase(vertexCount, fi);
  const nRaw: Vec3 = [y[nb], y[nb + 1], y[nb + 2]];
  const nLen = Math.max(1e-12, Math.hypot(nRaw[0], nRaw[1], nRaw[2]));
  const n: Vec3 = [nRaw[0] / nLen, nRaw[1] / nLen, nRaw[2] / nLen];
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

function denseGradientFromSparseRow(row: SparseRow, dim: number): number[] {
  const g = new Array<number>(dim).fill(0);
  for (let i = 0; i < row.idx.length; i++) g[row.idx[i]] += row.val[i];
  return g;
}

function addIncidenceEntries(entries: SymmetricEntry[], pair: VertexFaceIncidence, vertexCount: number) {
  const vb = 3 * pair.vi;
  const nb = lightNBase(vertexCount, pair.fi);
  for (let axis = 0; axis < 3; axis++) {
    const i = vb + axis;
    const j = nb + axis;
    entries.push({ i: Math.min(i, j), j: Math.max(i, j), value: 1 });
  }
}

function exactIncidenceConstraint(dim: number, pair: VertexFaceIncidence, vertexCount: number): QuadraticConstraint {
  const entries: SymmetricEntry[] = [];
  addIncidenceEntries(entries, pair, vertexCount);
  const b = new Array<number>(dim).fill(0);
  b[lightNBase(vertexCount, pair.fi) + 3] = -1;
  return {
    id: `inc:${pair.fi}:${pair.vi}`,
    sense: "eq",
    source: "exact",
    form: { dim, A: sparseSymmetricOperator(dim, entries), b, c: 0 },
  };
}

function exactUnitConstraint(dim: number, vertexCount: number, fi: number): QuadraticConstraint {
  const nb = lightNBase(vertexCount, fi);
  const entries: SymmetricEntry[] = [
    { i: nb, j: nb, value: 2 },
    { i: nb + 1, j: nb + 1, value: 2 },
    { i: nb + 2, j: nb + 2, value: 2 },
  ];
  return {
    id: `unit:${fi}`,
    sense: "eq",
    source: "exact",
    form: { dim, A: sparseSymmetricOperator(dim, entries), b: new Array<number>(dim).fill(0), c: -1 },
  };
}

function exactNonIncidenceConstraint(
  dim: number,
  pair: VertexFaceIncidence,
  vertexCount: number,
  margin: number
): QuadraticConstraint {
  const entries: SymmetricEntry[] = [];
  addIncidenceEntries(entries, pair, vertexCount);
  const b = new Array<number>(dim).fill(0);
  b[lightNBase(vertexCount, pair.fi) + 3] = -1;
  return {
    id: `convex:${pair.fi}:${pair.vi}`,
    sense: "le",
    source: "exact",
    form: { dim, A: sparseSymmetricOperator(dim, entries), b, c: margin },
  };
}

class FeasibilityProblemBuilder implements ProblemBuilder<void> {
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
    const state: PolyState = {
      vertices: new Array(this.vertexCount()).fill(0).map((_, i) => readLightVertex(y, i)),
      faces: this.faces.map((f) => [...f]),
      facePlanes: this.faces.map((_f, fi) => {
        const nb = lightNBase(this.vertexCount(), fi);
        return { n: [y[nb], y[nb + 1], y[nb + 2]] as Vec3, b: y[nb + 3] };
      }),
    };
    return buildPolyAuxState(state, this.topology);
  }

  private buildVolumeRow(y: ReadonlyArray<number>, params: FeasibilityOptimizeParams): SparseRow {
    const aux = this.buildAux(y);
    const row: SparseRow = { idx: [], val: [], c: aux.volume - params.volumeTarget };
    const eps = 1e-6;
    const yWork = y.slice();
    for (let k = 0; k < this.vertexDim(); k++) {
      const old = yWork[k];
      yWork[k] = old + eps;
      const vp = this.buildAux(yWork).volume;
      yWork[k] = old - eps;
      const vm = this.buildAux(yWork).volume;
      yWork[k] = old;
      const g = (vp - vm) / (2 * eps);
      if (!Number.isFinite(g) || Math.abs(g) < 1e-14) continue;
      row.idx.push(k);
      row.val.push(g);
    }
    return row;
  }

  private buildAntiRows(y: ReadonlyArray<number>, params: FeasibilityOptimizeParams): Array<{ id: string; row: SparseRow }> {
    const rows: Array<{ id: string; row: SparseRow }> = [];
    const targetFaces = this.activePieces.getTargetAntiFaces();
    const aux = this.buildAux(y);
    for (let i = 0; i < targetFaces.length; i++) {
      const fi = targetFaces[i];
      const edgeIdx = this.activePieces.getActiveEdge(fi);
      if (edgeIdx === undefined) continue;
      const g0 = antiResidual(fi, edgeIdx, y, this.faces, this.vertexCount(), aux, params.antiMargin);
      if (g0 <= 0) continue;

      const row: SparseRow = { idx: [], val: [], c: g0 };
      const eps = 1e-6;
      const yWork = y.slice();
      for (let k = 0; k < this.vertexDim(); k++) {
        const old = yWork[k];
        yWork[k] = old + eps;
        const gp = antiResidual(fi, edgeIdx, yWork, this.faces, this.vertexCount(), this.buildAux(yWork), params.antiMargin);
        yWork[k] = old - eps;
        const gm = antiResidual(fi, edgeIdx, yWork, this.faces, this.vertexCount(), this.buildAux(yWork), params.antiMargin);
        yWork[k] = old;
        const grad = (gp - gm) / (2 * eps);
        if (!Number.isFinite(grad) || Math.abs(grad) < 1e-14) continue;
        row.idx.push(k);
        row.val.push(grad);
      }
      rows.push({ id: `anti:${fi}:${edgeIdx}`, row });
    }
    return rows;
  }

  private moveObjectiveValue(y: ReadonlyArray<number>, moveW: number): number {
    let f = 0;
    for (let i = 0; i < this.baselineVertices.length; i++) {
      const b = 3 * i;
      const dx = y[b] - this.baselineVertices[i][0];
      const dy = y[b + 1] - this.baselineVertices[i][1];
      const dz = y[b + 2] - this.baselineVertices[i][2];
      f += moveW * (dx * dx + dy * dy + dz * dz);
    }
    return f;
  }

  buildProblem(xRef: ReadonlyArray<number>): OptimizationProblem {
    const params = this.paramsProvider();
    const dim = this.fullDim();
    const moveW = Math.max(0, params.moveWeight);

    const exactEq: QuadraticConstraint[] = [];
    for (let i = 0; i < this.topology.incidencePairs.length; i++) {
      exactEq.push(exactIncidenceConstraint(dim, this.topology.incidencePairs[i], this.vertexCount()));
    }
    for (let fi = 0; fi < this.faces.length; fi++) {
      exactEq.push(exactUnitConstraint(dim, this.vertexCount(), fi));
    }

    const volumeRow = this.buildVolumeRow(xRef, params);
    const localEq: QuadraticConstraint[] = [
      linearConstraintAsQuadratic(
        "volume",
        "eq",
        xRef,
        denseGradientFromSparseRow(volumeRow, dim),
        volumeRow.c,
        "local"
      ),
    ];

    const exactLe: QuadraticConstraint[] = [];
    for (let i = 0; i < this.topology.nonIncidencePairs.length; i++) {
      exactLe.push(
        exactNonIncidenceConstraint(
          dim,
          this.topology.nonIncidencePairs[i],
          this.vertexCount(),
          params.convexityMargin
        )
      );
    }

    const localLe: QuadraticConstraint[] = this.buildAntiRows(xRef, params).map((r) =>
      linearConstraintAsQuadratic(
        r.id,
        "le",
        xRef,
        denseGradientFromSparseRow(r.row, dim),
        r.row.c,
        "local"
      )
    );

    const metricEntries: SymmetricEntry[] = [];
    const metricB = new Array<number>(dim).fill(0);
    let metricC = 0;
    for (let i = 0; i < this.baselineVertices.length; i++) {
      const b = 3 * i;
      for (let axis = 0; axis < 3; axis++) {
        const idx = b + axis;
        metricEntries.push({ i: idx, j: idx, value: 2 * moveW });
        metricB[idx] = -2 * moveW * this.baselineVertices[i][axis];
        metricC += moveW * this.baselineVertices[i][axis] * this.baselineVertices[i][axis];
      }
    }

    return {
      dim,
      xRef: [...xRef],
      exactEq,
      exactLe,
      localEq,
      localLe,
      metric: {
        dim,
        A: sparseSymmetricOperator(dim, metricEntries),
        b: metricB,
        c: metricC,
      },
      objectiveValueOverride: (x: ReadonlyArray<number>) => this.moveObjectiveValue(x, moveW),
    };
  }
}

class FeasibilityOptimizerSession {
  private faces: number[][];
  private topology: PolyTopologyData;
  private params: FeasibilityOptimizeParams;
  private baselineVertices: Vec3[] = [];
  private targetAntiFaces: number[] = [];
  private activeEdgeByFace = new Map<number, number>();
  private dwellByFace = new Map<number, number>();
  private iter = 0;
  private lastDiag: FeasibilityOptimizeDiagnostics = {
    iter: 0,
    eqResidualL2: Number.POSITIVE_INFINITY,
    ineqViolationMax: Number.POSITIVE_INFINITY,
    volume: 0,
    activeConvexityCount: 0,
    activeAntiCount: 0,
  };

  private builder: FeasibilityProblemBuilder;
  private kernel: LinearizedAlmKernel;
  private state: OptimizerState;
  private problem: OptimizationProblem;

  constructor(state: PolyState, params?: Partial<FeasibilityOptimizeParams>) {
    const light = buildPolyLightModelFromState(state);
    this.faces = light.state.faces.map((f) => [...f]);
    this.topology = light.topology;
    this.params = { ...defaultParams, ...(params ?? {}) };
    this.params.stableFaceIndex = this.clampStableFace(this.params.stableFaceIndex);
    this.baselineVertices = light.state.vertices.map((p) => [p[0], p[1], p[2]] as Vec3);
    this.targetAntiFaces = this.faces.map((_f, fi) => fi).filter((fi) => fi !== this.params.stableFaceIndex);

    const y0 = packPolyLightState(light.state);
    this.refreshAntiPieces(y0, true);

    this.builder = new FeasibilityProblemBuilder(
      this.faces,
      this.topology,
      this.baselineVertices,
      () => this.params,
      {
        getTargetAntiFaces: () => this.targetAntiFaces,
        getActiveEdge: (fi: number) => this.activeEdgeByFace.get(fi),
      }
    );
    this.kernel = new LinearizedAlmKernel();
    this.problem = this.builder.buildProblem(y0);
    this.state = this.kernel.initialize(this.problem, y0, this.kernelParams());
    this.lastDiag = this.computeDiagnostics(this.state.x);
  }

  private kernelParams(): OptimizerHyperParams {
    return {
      rho: this.params.rho,
      tau: this.params.tau,
      cgIters: this.params.cgIters,
      cgTol: this.params.cgTol,
      lineSearchShrink: this.params.lineSearchShrink,
      lineSearchMaxSteps: this.params.lineSearchMaxSteps,
      adaptRho: this.params.adaptRho,
      rhoIncrease: this.params.rhoIncrease,
      rhoDecrease: this.params.rhoDecrease,
      rhoResidualRatio: this.params.rhoResidualRatio,
      rhoMin: this.params.rhoMin,
      rhoMax: this.params.rhoMax,
    };
  }

  private vertexCount(): number {
    return this.baselineVertices.length;
  }

  private clampStableFace(idx: number): number {
    if (this.faces.length === 0) return -1;
    if (idx < 0) return 0;
    if (idx >= this.faces.length) return this.faces.length - 1;
    return idx;
  }

  private buildAux(y: ReadonlyArray<number>) {
    const state: PolyState = {
      vertices: new Array(this.vertexCount()).fill(0).map((_, i) => readLightVertex(y, i)),
      faces: this.faces.map((f) => [...f]),
      facePlanes: this.faces.map((_f, fi) => {
        const nb = lightNBase(this.vertexCount(), fi);
        return { n: [y[nb], y[nb + 1], y[nb + 2]] as Vec3, b: y[nb + 3] };
      }),
    };
    return buildPolyAuxState(state, this.topology);
  }

  private minFaceMargin(fi: number, y: ReadonlyArray<number>, aux: ReturnType<typeof buildPolyAuxState>): { edge: number; margin: number } {
    const face = this.faces[fi];
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

  private refreshAntiPieces(y: ReadonlyArray<number>, force: boolean) {
    const aux = this.buildAux(y);
    for (const fi of this.targetAntiFaces) {
      const candidate = this.minFaceMargin(fi, y, aux);
      const curEdge = this.activeEdgeByFace.get(fi);
      const curDwell = this.dwellByFace.get(fi) ?? 0;
      if (force || curEdge === undefined) {
        this.activeEdgeByFace.set(fi, candidate.edge);
        this.dwellByFace.set(fi, 0);
        continue;
      }
      const curMargin = edgeMargin(fi, curEdge, y, this.faces, this.vertexCount(), aux);
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

  private computeEqResiduals(y: ReadonlyArray<number>): number[] {
    const out: number[] = [];
    for (let i = 0; i < this.topology.incidencePairs.length; i++) {
      out.push(incidenceConstraintValue(y, this.vertexCount(), this.topology.incidencePairs[i]));
    }
    for (let fi = 0; fi < this.faces.length; fi++) {
      const nb = lightNBase(this.vertexCount(), fi);
      const nx = y[nb];
      const ny = y[nb + 1];
      const nz = y[nb + 2];
      out.push(nx * nx + ny * ny + nz * nz - 1);
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

      this.refreshAntiPieces(this.state.x, false);
      this.problem = this.builder.buildProblem(this.state.x);
      this.kernel.rebindProblem(this.problem, this.state);
      const stats = this.kernel.step(this.problem, this.state, this.kernelParams(), 1);

      if (stats.accepted === 0 || stats.lastAlpha < this.params.minAcceptedAlpha) {
        this.lastDiag = this.computeDiagnostics(this.state.x);
        return this.shouldStop(this.lastDiag) || this.iter >= this.params.maxOuterIters;
      }

      this.params.rho = this.state.rho;
      this.lastDiag = this.computeDiagnostics(this.state.x);
      if (this.shouldStop(this.lastDiag)) return true;
    }
    return this.iter >= this.params.maxOuterIters || this.shouldStop(this.lastDiag);
  }

  getVertices(): Vec3[] {
    const out: Vec3[] = new Array(this.vertexCount());
    for (let i = 0; i < this.vertexCount(); i++) out[i] = readLightVertex(this.state.x, i);
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
