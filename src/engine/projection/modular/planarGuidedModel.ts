import type { Vec3 } from "../../math/types";
import { bestFitPlanePCA } from "../../geom/plane";
import {
  buildPolyTopology,
  lightFullDim,
  lightNBase,
  lightVertexDim,
  nonIncidenceConstraintValue,
  packPolyLightState,
  type PolyState,
  type VertexFaceIncidence,
} from "../../poly";
import {
  makeLocalQuadraticFromValueGradDiag,
  sparseSymmetricOperator,
} from "../../optimization/quadratic";
import type {
  OptimizationProblem,
  ProblemBuilder,
  QuadraticConstraint,
  SymmetricEntry,
} from "../../optimization/types";
import type { HandleSet } from "../index";
import { sumSquaredPlanarityResidual } from "../shared/metrics";
import { evaluateVertexTrackingObjectiveAndGradient } from "../shared/regularity";

type NonIncidence = VertexFaceIncidence & { di: number };

export type ModularConstraintMode = "inc_unit" | "inc_noninc_unit_squared_slack";

export type ModularGuidedALMParams = {
  rho: number;
  wFree: number;
  wHandle: number;
  lambdaReg: number;
  epsArea?: number;
  tau?: number;
  proxWeight?: number;
  cgIters?: number;
  cgTol?: number;
  lineSearchC1?: number;
  lineSearchShrink?: number;
  lineSearchMaxSteps?: number;
  adaptRho?: boolean;
  rhoIncrease?: number;
  rhoDecrease?: number;
  rhoResidualRatio?: number;
  rhoMin?: number;
  rhoMax?: number;
  normalProxWeight?: number;
  offsetProxWeight?: number;
  minAcceptedAlpha?: number;
  constraintTol?: number;
  constraintMode?: ModularConstraintMode;
};

function initializeSquaredSlackValues(
  baseY: ReadonlyArray<number>,
  vertexCount: number,
  nonIncidences: ReadonlyArray<NonIncidence>
): number[] {
  const d = new Array<number>(nonIncidences.length).fill(0);
  for (let i = 0; i < nonIncidences.length; i++) {
    const pair = nonIncidences[i];
    const gap = -nonIncidenceConstraintValue(baseY, vertexCount, pair);
    d[i] = Math.sqrt(Math.max(0, gap));
  }
  return d;
}

export function orientInitialNormalsOutward(
  faces: ReadonlyArray<ReadonlyArray<number>>,
  x: ReadonlyArray<Vec3>
): { normals: Vec3[]; offsets: number[] } {
  const center: Vec3 = [0, 0, 0];
  for (let i = 0; i < x.length; i++) {
    center[0] += x[i][0];
    center[1] += x[i][1];
    center[2] += x[i][2];
  }
  if (x.length > 0) {
    const inv = 1 / x.length;
    center[0] *= inv;
    center[1] *= inv;
    center[2] *= inv;
  }

  const normals: Vec3[] = new Array(faces.length);
  const offsets: number[] = new Array(faces.length);
  for (let fi = 0; fi < faces.length; fi++) {
    const pts = faces[fi].map((vi) => x[vi]);
    const plane = bestFitPlanePCA(pts);
    let n: Vec3 = [plane.n[0], plane.n[1], plane.n[2]];
    let b = plane.b;
    const side = n[0] * center[0] + n[1] * center[1] + n[2] * center[2] - b;
    if (side > 0) {
      n = [-n[0], -n[1], -n[2]];
      b = -b;
    }
    normals[fi] = n;
    offsets[fi] = b;
  }
  return { normals, offsets };
}

function countPlanarGuidedHardConstraints(
  faces: ReadonlyArray<ReadonlyArray<number>>,
  vertexCount: number,
  mode: ModularConstraintMode
): number {
  const topology = buildPolyTopology(faces, vertexCount);
  const mInc = topology.incidencePairs.length;
  if (mode === "inc_noninc_unit_squared_slack") return mInc + topology.nonIncidencePairs.length + faces.length;
  return mInc + faces.length;
}

export function packPlanarGuidedY(
  vertices: ReadonlyArray<Vec3>,
  normals: ReadonlyArray<Vec3>,
  offsets: ReadonlyArray<number>,
  mode: ModularConstraintMode,
  faces: ReadonlyArray<ReadonlyArray<number>>,
  initialD?: ReadonlyArray<number>
): number[] {
  const baseState: PolyState = {
    vertices: vertices.map((p) => [p[0], p[1], p[2]]),
    faces: faces.map((f) => [...f]),
    facePlanes: normals.map((n, fi) => ({ n: [n[0], n[1], n[2]], b: offsets[fi] })),
  };
  const base = packPolyLightState(baseState);
  if (mode !== "inc_noninc_unit_squared_slack") return base;

  const nonInc = buildPolyTopology(faces, vertices.length).nonIncidencePairs.map((pair, di) => ({ ...pair, di }));
  const dVals = initialD ? [...initialD] : initializeSquaredSlackValues(base, vertices.length, nonInc);

  const out = new Array<number>(base.length + nonInc.length);
  for (let i = 0; i < base.length; i++) out[i] = base[i];
  for (let i = 0; i < nonInc.length; i++) out[base.length + i] = dVals[i] ?? 0;
  return out;
}

export function unpackPlanarGuidedY(
  y: ReadonlyArray<number>,
  vertexCount: number,
  faceCount: number,
  mode: ModularConstraintMode,
  faces: ReadonlyArray<ReadonlyArray<number>>
): { vertices: Vec3[]; normals: Vec3[]; offsets: number[]; d: number[] } {
  const vertices: Vec3[] = new Array(vertexCount);
  const normals: Vec3[] = new Array(faceCount);
  const offsets: number[] = new Array(faceCount);
  for (let i = 0; i < vertexCount; i++) {
    const b = 3 * i;
    vertices[i] = [y[b], y[b + 1], y[b + 2]];
  }
  for (let fi = 0; fi < faceCount; fi++) {
    const nb = lightNBase(vertexCount, fi);
    normals[fi] = [y[nb], y[nb + 1], y[nb + 2]];
    offsets[fi] = y[nb + 3];
  }
  const d: number[] = [];
  if (mode === "inc_noninc_unit_squared_slack") {
    const nonInc = buildPolyTopology(faces, vertexCount).nonIncidencePairs;
    const base = lightFullDim(vertexCount, faceCount);
    for (let i = 0; i < nonInc.length; i++) d.push(y[base + i] ?? 0);
  }
  return { vertices, normals, offsets, d };
}

export function computeTotalPlanarityViolation(
  faces: ReadonlyArray<ReadonlyArray<number>>,
  positions: ReadonlyArray<Vec3>
): number {
  return sumSquaredPlanarityResidual(faces, positions);
}

function addIncidenceEntries(entries: SymmetricEntry[], vi: number, vertexCount: number, fi: number) {
  const vb = 3 * vi;
  const nb = lightNBase(vertexCount, fi);
  for (let axis = 0; axis < 3; axis++) {
    const i = vb + axis;
    const j = nb + axis;
    entries.push({ i: Math.min(i, j), j: Math.max(i, j), value: 1 });
  }
}

function incidenceConstraint(
  dim: number,
  vertexCount: number,
  pair: VertexFaceIncidence
): QuadraticConstraint {
  const entries: SymmetricEntry[] = [];
  addIncidenceEntries(entries, pair.vi, vertexCount, pair.fi);
  const b = new Array<number>(dim).fill(0);
  b[lightNBase(vertexCount, pair.fi) + 3] = -1;
  return {
    id: `inc:${pair.fi}:${pair.vi}`,
    sense: "eq",
    source: "exact",
    form: { dim, A: sparseSymmetricOperator(dim, entries), b, c: 0 },
  };
}

function squaredSlackNonIncidenceConstraint(
  dim: number,
  vertexCount: number,
  pair: NonIncidence,
  dIndex: number
): QuadraticConstraint {
  const entries: SymmetricEntry[] = [{ i: dIndex, j: dIndex, value: 2 }];
  addIncidenceEntries(entries, pair.vi, vertexCount, pair.fi);
  const b = new Array<number>(dim).fill(0);
  b[lightNBase(vertexCount, pair.fi) + 3] = -1;
  return {
    id: `noninc_sq_slack:${pair.fi}:${pair.vi}:${pair.di}`,
    sense: "eq",
    source: "exact",
    form: { dim, A: sparseSymmetricOperator(dim, entries), b, c: 0 },
  };
}

function unitNormalConstraint(dim: number, vertexCount: number, fi: number): QuadraticConstraint {
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

export class PlanarGuidedProblemBuilder implements ProblemBuilder<void> {
  private faces: number[][];
  private topology = buildPolyTopology([], 0);
  private x0: Vec3[];
  private params: ModularGuidedALMParams;
  private handles: HandleSet;
  private yRefProvider: () => ReadonlyArray<number>;
  private mode: ModularConstraintMode;
  private incidences: VertexFaceIncidence[] = [];
  private nonIncidences: NonIncidence[] = [];

  constructor(
    faces: number[][],
    x0: Vec3[],
    params: ModularGuidedALMParams,
    handles: HandleSet,
    yRefProvider: () => ReadonlyArray<number>,
    mode: ModularConstraintMode
  ) {
    this.faces = faces.map((f) => [...f]);
    this.topology = buildPolyTopology(this.faces, x0.length);
    this.x0 = x0.map((p) => [p[0], p[1], p[2]]);
    this.params = { ...params };
    this.handles = handles;
    this.yRefProvider = yRefProvider;
    this.mode = mode;
    this.buildPairs();
  }

  setParams(next: Partial<ModularGuidedALMParams>) {
    this.params = { ...this.params, ...next };
  }

  setHandles(handles: HandleSet) {
    this.handles = handles;
  }

  setBaseline(x0: Vec3[]) {
    this.x0 = x0.map((p) => [p[0], p[1], p[2]]);
  }

  hardConstraintCount(): number {
    return countPlanarGuidedHardConstraints(this.faces, this.x0.length, this.mode);
  }

  initializeContext(): void {
    return;
  }

  private buildPairs() {
    this.incidences = this.topology.incidencePairs.map((pair) => ({ fi: pair.fi, vi: pair.vi }));
    this.nonIncidences = this.mode === "inc_noninc_unit_squared_slack"
      ? this.topology.nonIncidencePairs.map((pair, di) => ({ fi: pair.fi, vi: pair.vi, di }))
      : [];
  }

  private vertexDim(): number {
    return lightVertexDim(this.x0.length);
  }

  private dBase(): number {
    return lightFullDim(this.x0.length, this.faces.length);
  }

  private fullDim(): number {
    return this.dBase() + this.nonIncidences.length;
  }

  private dIndex(di: number): number {
    return this.dBase() + di;
  }

  private objectiveAndGradient(y: ReadonlyArray<number>, gradOut?: number[]): number {
    const f = evaluateVertexTrackingObjectiveAndGradient(
      y,
      {
        baseline: this.x0,
        handles: this.handles.targets,
        wFree: this.params.wFree,
        wHandle: this.params.wHandle,
        lambdaReg: Math.max(0, this.params.lambdaReg),
        epsArea: Math.max(1e-12, this.params.epsArea ?? 1e-8),
        faces: this.faces,
      },
      gradOut
    );
    if (gradOut) {
      for (let i = this.vertexDim(); i < gradOut.length; i++) gradOut[i] = 0;
    }
    return f;
  }

  private buildObjectiveQuadratic(y: ReadonlyArray<number>) {
    const yRef = this.yRefProvider();
    const dim = this.fullDim();
    const tau = Math.max(1e-10, this.params.tau ?? 1e-6);
    const proxWeight = Math.max(0, this.params.proxWeight ?? 0);
    const normalProxWeight = Math.max(0, this.params.normalProxWeight ?? 1);
    const offsetProxWeight = Math.max(0, this.params.offsetProxWeight ?? 1);

    const gradient = new Array<number>(dim).fill(0);
    let value = this.objectiveAndGradient(y, gradient);

    if (proxWeight > 0) {
      let proxV = 0;
      for (let i = 0; i < this.x0.length; i++) {
        const b = 3 * i;
        const dx = y[b] - this.x0[i][0];
        const dy = y[b + 1] - this.x0[i][1];
        const dz = y[b + 2] - this.x0[i][2];
        gradient[b] += proxWeight * dx;
        gradient[b + 1] += proxWeight * dy;
        gradient[b + 2] += proxWeight * dz;
        proxV += dx * dx + dy * dy + dz * dz;
      }
      value += 0.5 * proxWeight * proxV;
    }

    if (normalProxWeight > 0 || offsetProxWeight > 0) {
      let proxNB = 0;
      for (let fi = 0; fi < this.faces.length; fi++) {
        const nb = lightNBase(this.x0.length, fi);
        const dnx = y[nb] - yRef[nb];
        const dny = y[nb + 1] - yRef[nb + 1];
        const dnz = y[nb + 2] - yRef[nb + 2];
        const db = y[nb + 3] - yRef[nb + 3];
        gradient[nb] += normalProxWeight * dnx;
        gradient[nb + 1] += normalProxWeight * dny;
        gradient[nb + 2] += normalProxWeight * dnz;
        gradient[nb + 3] += offsetProxWeight * db;
        proxNB += normalProxWeight * (dnx * dnx + dny * dny + dnz * dnz);
        proxNB += offsetProxWeight * db * db;
      }
      value += 0.5 * proxNB;
    }

    const hDiag = new Array<number>(dim).fill(tau);
    for (let i = 0; i < this.x0.length; i++) {
      const b = 3 * i;
      const isHandle = this.handles.targets.has(i);
      const w = isHandle ? this.params.wHandle : this.params.wFree;
      const val = 2 * w + proxWeight + tau;
      hDiag[b] = val;
      hDiag[b + 1] = val;
      hDiag[b + 2] = val;
    }
    for (let fi = 0; fi < this.faces.length; fi++) {
      const nb = lightNBase(this.x0.length, fi);
      hDiag[nb] += normalProxWeight;
      hDiag[nb + 1] += normalProxWeight;
      hDiag[nb + 2] += normalProxWeight;
      hDiag[nb + 3] += offsetProxWeight;
    }

    return {
      metric: makeLocalQuadraticFromValueGradDiag(y, value, gradient, hDiag),
      objectiveValueOverride: (x: ReadonlyArray<number>) => {
        const base = this.objectiveAndGradient(x);
        let proxV = 0;
        if (proxWeight > 0) {
          for (let i = 0; i < this.x0.length; i++) {
            const b = 3 * i;
            const dx = x[b] - this.x0[i][0];
            const dy = x[b + 1] - this.x0[i][1];
            const dz = x[b + 2] - this.x0[i][2];
            proxV += dx * dx + dy * dy + dz * dz;
          }
        }
        let proxNB = 0;
        if (normalProxWeight > 0 || offsetProxWeight > 0) {
          for (let fi = 0; fi < this.faces.length; fi++) {
            const nb = lightNBase(this.x0.length, fi);
            const dnx = x[nb] - yRef[nb];
            const dny = x[nb + 1] - yRef[nb + 1];
            const dnz = x[nb + 2] - yRef[nb + 2];
            const db = x[nb + 3] - yRef[nb + 3];
            proxNB += normalProxWeight * (dnx * dnx + dny * dny + dnz * dnz);
            proxNB += offsetProxWeight * db * db;
          }
        }
        return base + 0.5 * proxWeight * proxV + 0.5 * proxNB;
      },
    };
  }

  buildProblem(xRef: ReadonlyArray<number>): OptimizationProblem {
    const dim = this.fullDim();
    const exactEq: QuadraticConstraint[] = [];

    for (let i = 0; i < this.incidences.length; i++) {
      exactEq.push(incidenceConstraint(dim, this.x0.length, this.incidences[i]));
    }
    for (let i = 0; i < this.nonIncidences.length; i++) {
      const p = this.nonIncidences[i];
      const dIndex = this.dIndex(p.di);
      exactEq.push(squaredSlackNonIncidenceConstraint(dim, this.x0.length, p, dIndex));
    }
    for (let fi = 0; fi < this.faces.length; fi++) {
      exactEq.push(unitNormalConstraint(dim, this.x0.length, fi));
    }

    const objective = this.buildObjectiveQuadratic(xRef);

    return {
      dim,
      xRef: [...xRef],
      exactEq,
      exactLe: [],
      localEq: [],
      localLe: [],
      metric: objective.metric,
      objectiveValueOverride: objective.objectiveValueOverride,
    };
  }
}
