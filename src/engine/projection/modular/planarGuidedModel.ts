import type { Vec3 } from "../../math/types";
import { bestFitPlanePCA } from "../../geom/plane";
import {
  buildPolyTopology,
  lightBIndex,
  lightFullDim,
  lightNBase,
  lightVertexDim,
  nonIncidenceConstraintValue,
  packPolyLightState,
  incidenceConstraintLinearization,
  incidenceConstraintValue,
  squaredSlackNonIncidenceConstraintLinearization,
  unitNormalConstraintLinearization,
  unitNormalConstraintValue,
  type PolyState,
  type VertexFaceIncidence,
} from "../../poly";
import type { HandleSet } from "../index";
import { sumSquaredPlanarityResidual } from "../shared/metrics";
import { evaluateVertexTrackingObjectiveAndGradient } from "../shared/regularity";
import type { ConstraintLinearization, MetaModel, MetaModelBuilder, MetaState } from "./types";

type NonIncidence = VertexFaceIncidence & {
  di: number;
};

type IncLinearized = {
  kind: "inc";
  fi: number;
  vi: number;
  gV: Vec3;
  gN: Vec3;
  gB: number;
};

type NonIncLinearized = {
  kind: "noninc";
  fi: number;
  vi: number;
  di: number;
  gV: Vec3;
  gN: Vec3;
  gB: number;
  gD: number;
};

type UnitLinearized = {
  kind: "unit";
  fi: number;
  gN: Vec3;
};

type LinearizedRow = IncLinearized | NonIncLinearized | UnitLinearized;

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

export function orientInitialNormalsOutward(faces: ReadonlyArray<ReadonlyArray<number>>, x: ReadonlyArray<Vec3>): { normals: Vec3[]; offsets: number[] } {
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

export function countPlanarGuidedHardConstraints(
  faces: ReadonlyArray<ReadonlyArray<number>>,
  vertexCount: number,
  mode: ModularConstraintMode
): number {
  const topology = buildPolyTopology(faces, vertexCount);
  const mInc = topology.incidencePairs.length;
  if (mode === "inc_noninc_unit_squared_slack") {
    return mInc + topology.nonIncidencePairs.length + faces.length;
  }
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
  const dVals = initialD
    ? [...initialD]
    : initializeSquaredSlackValues(base, vertices.length, nonInc);

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

export class PlanarGuidedModelBuilder implements MetaModelBuilder {
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

  private linearizeConstraints(y: ReadonlyArray<number>): ConstraintLinearization {
    const mInc = this.incidences.length;
    const mNon = this.nonIncidences.length;
    const rows: LinearizedRow[] = new Array(mInc + mNon + this.faces.length);
    const c0 = new Array<number>(rows.length);

        for (let ri = 0; ri < mInc; ri++) {
      const pair = this.incidences[ri];
      const lin = incidenceConstraintLinearization(y, this.x0.length, pair);
      rows[ri] = {
        kind: "inc",
        fi: pair.fi,
        vi: pair.vi,
        gV: lin.gV,
        gN: lin.gN,
        gB: lin.gB,
      };
      c0[ri] = lin.value;
    }

    for (let qi = 0; qi < mNon; qi++) {
      const rowIndex = mInc + qi;
      const pair = this.nonIncidences[qi];
      const d = y[this.dIndex(pair.di)];
      const lin = squaredSlackNonIncidenceConstraintLinearization(y, this.x0.length, pair, d);
      rows[rowIndex] = {
        kind: "noninc",
        fi: pair.fi,
        vi: pair.vi,
        di: pair.di,
        gV: lin.gV,
        gN: lin.gN,
        gB: lin.gB,
        gD: lin.gD,
      };
      c0[rowIndex] = lin.value;
    }

    for (let fi = 0; fi < this.faces.length; fi++) {
      const idx = mInc + mNon + fi;
      const lin = unitNormalConstraintLinearization(y, this.x0.length, fi);
      rows[idx] = { kind: "unit", fi, gN: lin.gN };
      c0[idx] = lin.value;
    }

    const applyJ = (v: ReadonlyArray<number>): number[] => {
      const out = new Array<number>(rows.length).fill(0);
      for (let i = 0; i < rows.length; i++) {
        const r = rows[i];
        if (r.kind === "inc") {
          const vb = 3 * r.vi;
          const nb = lightNBase(this.x0.length, r.fi);
          out[i] =
            r.gV[0] * v[vb] + r.gV[1] * v[vb + 1] + r.gV[2] * v[vb + 2] +
            r.gN[0] * v[nb] + r.gN[1] * v[nb + 1] + r.gN[2] * v[nb + 2] +
            r.gB * v[lightBIndex(this.x0.length, r.fi)];
        } else if (r.kind === "noninc") {
          const vb = 3 * r.vi;
          const nb = lightNBase(this.x0.length, r.fi);
          out[i] =
            r.gV[0] * v[vb] + r.gV[1] * v[vb + 1] + r.gV[2] * v[vb + 2] +
            r.gN[0] * v[nb] + r.gN[1] * v[nb + 1] + r.gN[2] * v[nb + 2] +
            r.gB * v[lightBIndex(this.x0.length, r.fi)] +
            r.gD * v[this.dIndex(r.di)];
        } else {
          const nb = lightNBase(this.x0.length, r.fi);
          out[i] = r.gN[0] * v[nb] + r.gN[1] * v[nb + 1] + r.gN[2] * v[nb + 2];
        }
      }
      return out;
    };

    const applyJT = (w: ReadonlyArray<number>): number[] => {
      const out = new Array<number>(this.fullDim()).fill(0);
      for (let i = 0; i < rows.length; i++) {
        const wi = w[i];
        const r = rows[i];
        if (r.kind === "inc") {
          const vb = 3 * r.vi;
          const nb = lightNBase(this.x0.length, r.fi);
          const bb = lightBIndex(this.x0.length, r.fi);
          out[vb] += wi * r.gV[0];
          out[vb + 1] += wi * r.gV[1];
          out[vb + 2] += wi * r.gV[2];
          out[nb] += wi * r.gN[0];
          out[nb + 1] += wi * r.gN[1];
          out[nb + 2] += wi * r.gN[2];
          out[bb] += wi * r.gB;
        } else if (r.kind === "noninc") {
          const vb = 3 * r.vi;
          const nb = lightNBase(this.x0.length, r.fi);
          const bb = lightBIndex(this.x0.length, r.fi);
          out[vb] += wi * r.gV[0];
          out[vb + 1] += wi * r.gV[1];
          out[vb + 2] += wi * r.gV[2];
          out[nb] += wi * r.gN[0];
          out[nb + 1] += wi * r.gN[1];
          out[nb + 2] += wi * r.gN[2];
          out[bb] += wi * r.gB;
          out[this.dIndex(r.di)] += wi * r.gD;
        } else {
          const nb = lightNBase(this.x0.length, r.fi);
          out[nb] += wi * r.gN[0];
          out[nb + 1] += wi * r.gN[1];
          out[nb + 2] += wi * r.gN[2];
        }
      }
      return out;
    };

    return { c0, applyJ, applyJT };
  }

  private evalConstraintsOnly(y: ReadonlyArray<number>): number[] {
    const mInc = this.incidences.length;
    const mNon = this.nonIncidences.length;
    const out = new Array<number>(mInc + mNon + this.faces.length);
    for (let ri = 0; ri < mInc; ri++) {
      out[ri] = incidenceConstraintValue(y, this.x0.length, this.incidences[ri]);
    }
    for (let qi = 0; qi < mNon; qi++) {
      const rowIndex = mInc + qi;
      const pair = this.nonIncidences[qi];
      const d = y[this.dIndex(pair.di)];
      out[rowIndex] = squaredSlackNonIncidenceConstraintLinearization(y, this.x0.length, pair, d).value;
    }
    for (let fi = 0; fi < this.faces.length; fi++) {
      out[mInc + mNon + fi] = unitNormalConstraintValue(y, this.x0.length, fi);
    }
    return out;
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

  build(_state: Readonly<MetaState>): MetaModel {
    const y = _state.y;
    const yRef = this.yRefProvider();
    const dim = this.fullDim();
    const tau = Math.max(1e-10, this.params.tau ?? 1e-6);
    const proxWeight = Math.max(0, this.params.proxWeight ?? 0);
    const normalProxWeight = Math.max(0, this.params.normalProxWeight ?? 1);
    const offsetProxWeight = Math.max(0, this.params.offsetProxWeight ?? 1);

    const gradient = new Array<number>(dim).fill(0);
    this.objectiveAndGradient(y, gradient);

    if (proxWeight > 0) {
      for (let i = 0; i < this.x0.length; i++) {
        const b = 3 * i;
        gradient[b] += proxWeight * (y[b] - this.x0[i][0]);
        gradient[b + 1] += proxWeight * (y[b + 1] - this.x0[i][1]);
        gradient[b + 2] += proxWeight * (y[b + 2] - this.x0[i][2]);
      }
    }

    if (normalProxWeight > 0 || offsetProxWeight > 0) {
      for (let fi = 0; fi < this.faces.length; fi++) {
        const nb = lightNBase(this.x0.length, fi);
        gradient[nb] += normalProxWeight * (y[nb] - yRef[nb]);
        gradient[nb + 1] += normalProxWeight * (y[nb + 1] - yRef[nb + 1]);
        gradient[nb + 2] += normalProxWeight * (y[nb + 2] - yRef[nb + 2]);
        gradient[nb + 3] += offsetProxWeight * (y[nb + 3] - yRef[nb + 3]);
      }
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

    const linearization = this.linearizeConstraints(y);
    return {
      dim,
      gradient,
      hDiag,
      hard: {
        linearization,
        evaluate: (yy: ReadonlyArray<number>) => this.evalConstraintsOnly(yy),
      },
      merit: (yy: ReadonlyArray<number>, u: ReadonlyArray<number>, rho: number) => {
        const f = this.objectiveAndGradient(yy);
        const c = this.evalConstraintsOnly(yy);
        let pen = 0;
        for (let i = 0; i < c.length; i++) {
          const t = c[i] + u[i];
          pen += t * t;
        }

        let proxV = 0;
        if (proxWeight > 0) {
          for (let i = 0; i < this.x0.length; i++) {
            const b = 3 * i;
            const dx = yy[b] - this.x0[i][0];
            const dy = yy[b + 1] - this.x0[i][1];
            const dz = yy[b + 2] - this.x0[i][2];
            proxV += dx * dx + dy * dy + dz * dz;
          }
        }

        let proxNB = 0;
        if (normalProxWeight > 0 || offsetProxWeight > 0) {
          for (let fi = 0; fi < this.faces.length; fi++) {
            const nb = lightNBase(this.x0.length, fi);
            const dnx = yy[nb] - yRef[nb];
            const dny = yy[nb + 1] - yRef[nb + 1];
            const dnz = yy[nb + 2] - yRef[nb + 2];
            const db = yy[nb + 3] - yRef[nb + 3];
            proxNB += normalProxWeight * (dnx * dnx + dny * dny + dnz * dnz);
            proxNB += offsetProxWeight * db * db;
          }
        }

        return f + 0.5 * rho * pen + 0.5 * proxWeight * proxV + 0.5 * proxNB;
      },
    };
  }
}




