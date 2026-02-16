import type { Vec3 } from "../math/types";
import { bestFitPlanePCA } from "../geom/plane";
import type { HandleSet, IProjector } from "./index";
import { sumSquaredPlanarityResidual } from "./shared/metrics";
import { dotN, normN, solveCG } from "./shared/numeric";
import { evaluateVertexTrackingObjectiveAndGradient } from "./shared/regularity";

type Incidence = {
  fi: number;
  vi: number;
};

type NonIncidence = {
  fi: number;
  vi: number;
};

type IncLinearized = {
  kind: "inc";
  fi: number;
  vi: number;
  c0: number;
  gV: Vec3;
  gN: Vec3;
  gB: number;
};

type NonIncLinearized = {
  kind: "noninc";
  fi: number;
  vi: number;
  di: number;
  c0: number;
  gV: Vec3;
  gN: Vec3;
  gB: number;
  gD: number;
};

type UnitLinearized = {
  kind: "unit";
  fi: number;
  c0: number;
  gN: Vec3;
};

type LinearizedRow = IncLinearized | NonIncLinearized | UnitLinearized;

export type GuidedALMSquaredSlackParams = {
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
};

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export class GuidedALMSquaredSlackPlanarProjector implements IProjector {
  private faces: number[][];
  private x0: Vec3[] = [];
  private x: Vec3[] = [];
  private normals: Vec3[] = [];
  private offsets: number[] = [];
  private dVals: number[] = [];
  private params: GuidedALMSquaredSlackParams;
  private handles: HandleSet = { targets: new Map() };
  private lastTotalViolation = 0;

  // One scaled multiplier per hard constraint:
  // [all incidences n_f^T v_i-b_f=0, then all non-incidence n_f^T v_i-b_f=d^2,
  //  then all unit constraints n_f^T n_f-1=0]
  private u: number[] = [];
  private incidences: Incidence[] = [];
  private nonIncidences: NonIncidence[] = [];

  constructor(faces: number[][], x0: Vec3[], params: GuidedALMSquaredSlackParams) {
    this.faces = faces.map((f) => [...f]);
    this.params = { ...params };
    this.reset(x0);
  }

  private vertexDim(): number {
    return this.x.length * 3;
  }

  private dBase(): number {
    return this.vertexDim() + 4 * this.faces.length;
  }

  private fullDim(): number {
    return this.dBase() + this.dVals.length;
  }

  private nBase(fi: number): number {
    return this.vertexDim() + 4 * fi;
  }

  private bIndex(fi: number): number {
    return this.vertexDim() + 4 * fi + 3;
  }

  private dIndex(di: number): number {
    return this.dBase() + di;
  }

  private buildPairLists() {
    this.incidences = [];
    this.nonIncidences = [];

    const faceSets = this.faces.map((f) => new Set<number>(f));
    for (let fi = 0; fi < this.faces.length; fi++) {
      for (const vi of this.faces[fi]) this.incidences.push({ fi, vi });
    }
    for (let fi = 0; fi < this.faces.length; fi++) {
      const s = faceSets[fi];
      for (let vi = 0; vi < this.x.length; vi++) {
        if (!s.has(vi)) this.nonIncidences.push({ fi, vi });
      }
    }
  }

  private orientInitialNormalsInward() {
    const center: Vec3 = [0, 0, 0];
    for (let i = 0; i < this.x.length; i++) {
      center[0] += this.x[i][0];
      center[1] += this.x[i][1];
      center[2] += this.x[i][2];
    }
    if (this.x.length > 0) {
      const inv = 1 / this.x.length;
      center[0] *= inv;
      center[1] *= inv;
      center[2] *= inv;
    }

    this.normals = new Array(this.faces.length);
    this.offsets = new Array(this.faces.length);

    for (let fi = 0; fi < this.faces.length; fi++) {
      const pts = this.faces[fi].map((vi) => this.x[vi]);
      const plane = bestFitPlanePCA(pts);
      let n: Vec3 = [plane.n[0], plane.n[1], plane.n[2]];
      let b = plane.b;
      // For inward normals of a convex polyhedron, interior point should satisfy n^T x >= b.
      const side = n[0] * center[0] + n[1] * center[1] + n[2] * center[2] - b;
      if (side < 0) {
        n = [-n[0], -n[1], -n[2]];
        b = -b;
      }
      this.normals[fi] = n;
      this.offsets[fi] = b;
    }
  }

  private initializeSlacksFromCurrentState() {
    this.dVals = new Array(this.nonIncidences.length).fill(0);
    for (let di = 0; di < this.nonIncidences.length; di++) {
      const { fi, vi } = this.nonIncidences[di];
      const n = this.normals[fi];
      const v = this.x[vi];
      const b = this.offsets[fi];
      const s = dot3(n, v) - b;
      this.dVals[di] = Math.sqrt(Math.max(0, s));
    }
  }

  reset(x0: Vec3[]) {
    this.x0 = x0.map((p) => [p[0], p[1], p[2]] as Vec3);
    this.x = x0.map((p) => [p[0], p[1], p[2]] as Vec3);

    this.buildPairLists();
    this.orientInitialNormalsInward();
    this.initializeSlacksFromCurrentState();

    const m = this.incidences.length + this.nonIncidences.length + this.faces.length;
    this.u = new Array(m).fill(0);

    this.lastTotalViolation = this.computeTotalPlanarityViolation(this.x);
  }

  setHandles(handles: HandleSet) {
    this.handles = handles;
  }

  setParams(next: Partial<GuidedALMSquaredSlackParams>) {
    this.params = { ...this.params, ...next };
  }

  private flattenY(): number[] {
    const out = new Array<number>(this.fullDim());
    for (let i = 0; i < this.x.length; i++) {
      const b = 3 * i;
      out[b] = this.x[i][0];
      out[b + 1] = this.x[i][1];
      out[b + 2] = this.x[i][2];
    }
    for (let fi = 0; fi < this.faces.length; fi++) {
      const nb = this.nBase(fi);
      out[nb] = this.normals[fi][0];
      out[nb + 1] = this.normals[fi][1];
      out[nb + 2] = this.normals[fi][2];
      out[nb + 3] = this.offsets[fi];
    }
    for (let di = 0; di < this.dVals.length; di++) {
      out[this.dIndex(di)] = this.dVals[di];
    }
    return out;
  }

  private syncFromY(y: ReadonlyArray<number>) {
    for (let i = 0; i < this.x.length; i++) {
      const b = 3 * i;
      this.x[i][0] = y[b];
      this.x[i][1] = y[b + 1];
      this.x[i][2] = y[b + 2];
    }
    for (let fi = 0; fi < this.faces.length; fi++) {
      const nb = this.nBase(fi);
      this.normals[fi][0] = y[nb];
      this.normals[fi][1] = y[nb + 1];
      this.normals[fi][2] = y[nb + 2];
      this.offsets[fi] = y[nb + 3];
    }
    for (let di = 0; di < this.dVals.length; di++) {
      this.dVals[di] = y[this.dIndex(di)];
    }
  }

  private linearizeConstraints(y: ReadonlyArray<number>): { rows: LinearizedRow[]; c0: number[] } {
    const mInc = this.incidences.length;
    const mNon = this.nonIncidences.length;
    const m = mInc + mNon + this.faces.length;
    const rows: LinearizedRow[] = new Array(m);
    const c0 = new Array<number>(m);

    // n_f^T v_i - b_f = 0 for incident pairs.
    for (let ri = 0; ri < mInc; ri++) {
      const { fi, vi } = this.incidences[ri];
      const vBase = 3 * vi;
      const nBase = this.nBase(fi);
      const v: Vec3 = [y[vBase], y[vBase + 1], y[vBase + 2]];
      const n: Vec3 = [y[nBase], y[nBase + 1], y[nBase + 2]];
      const b = y[this.bIndex(fi)];

      const c = dot3(n, v) - b;
      rows[ri] = { kind: "inc", fi, vi, c0: c, gV: [n[0], n[1], n[2]], gN: [v[0], v[1], v[2]], gB: -1 };
      c0[ri] = c;
    }

    // n_f^T v_i - b_f = d^2 for non-incident pairs.
    for (let qi = 0; qi < mNon; qi++) {
      const rowIndex = mInc + qi;
      const { fi, vi } = this.nonIncidences[qi];
      const vBase = 3 * vi;
      const nBase = this.nBase(fi);
      const dIndex = this.dIndex(qi);
      const v: Vec3 = [y[vBase], y[vBase + 1], y[vBase + 2]];
      const n: Vec3 = [y[nBase], y[nBase + 1], y[nBase + 2]];
      const b = y[this.bIndex(fi)];
      const d = y[dIndex];

      const c = dot3(n, v) - b - d * d;
      rows[rowIndex] = {
        kind: "noninc",
        fi,
        vi,
        di: qi,
        c0: c,
        gV: [n[0], n[1], n[2]],
        gN: [v[0], v[1], v[2]],
        gB: -1,
        gD: -2 * d,
      };
      c0[rowIndex] = c;
    }

    // n_f^T n_f - 1 = 0.
    for (let fi = 0; fi < this.faces.length; fi++) {
      const rowIndex = mInc + mNon + fi;
      const nBase = this.nBase(fi);
      const n: Vec3 = [y[nBase], y[nBase + 1], y[nBase + 2]];
      const c = dot3(n, n) - 1;
      rows[rowIndex] = { kind: "unit", fi, c0: c, gN: [2 * n[0], 2 * n[1], 2 * n[2]] };
      c0[rowIndex] = c;
    }

    return { rows, c0 };
  }

  private evalConstraintsOnly(y: ReadonlyArray<number>): number[] {
    const mInc = this.incidences.length;
    const mNon = this.nonIncidences.length;
    const out = new Array<number>(mInc + mNon + this.faces.length);

    for (let ri = 0; ri < mInc; ri++) {
      const { fi, vi } = this.incidences[ri];
      const vBase = 3 * vi;
      const nBase = this.nBase(fi);
      const v: Vec3 = [y[vBase], y[vBase + 1], y[vBase + 2]];
      const n: Vec3 = [y[nBase], y[nBase + 1], y[nBase + 2]];
      const b = y[this.bIndex(fi)];
      out[ri] = dot3(n, v) - b;
    }

    for (let qi = 0; qi < mNon; qi++) {
      const rowIndex = mInc + qi;
      const { fi, vi } = this.nonIncidences[qi];
      const vBase = 3 * vi;
      const nBase = this.nBase(fi);
      const dIndex = this.dIndex(qi);
      const v: Vec3 = [y[vBase], y[vBase + 1], y[vBase + 2]];
      const n: Vec3 = [y[nBase], y[nBase + 1], y[nBase + 2]];
      const b = y[this.bIndex(fi)];
      const d = y[dIndex];
      out[rowIndex] = dot3(n, v) - b - d * d;
    }

    for (let fi = 0; fi < this.faces.length; fi++) {
      const nBase = this.nBase(fi);
      const n: Vec3 = [y[nBase], y[nBase + 1], y[nBase + 2]];
      out[mInc + mNon + fi] = dot3(n, n) - 1;
    }
    return out;
  }

  private jTimesVector(rows: ReadonlyArray<LinearizedRow>, v: ReadonlyArray<number>): number[] {
    const out = new Array<number>(rows.length).fill(0);
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      if (r.kind === "inc") {
        const vb = 3 * r.vi;
        const nb = this.nBase(r.fi);
        out[i] =
          r.gV[0] * v[vb] + r.gV[1] * v[vb + 1] + r.gV[2] * v[vb + 2] +
          r.gN[0] * v[nb] + r.gN[1] * v[nb + 1] + r.gN[2] * v[nb + 2] +
          r.gB * v[nb + 3];
      } else if (r.kind === "noninc") {
        const vb = 3 * r.vi;
        const nb = this.nBase(r.fi);
        const di = this.dIndex(r.di);
        out[i] =
          r.gV[0] * v[vb] + r.gV[1] * v[vb + 1] + r.gV[2] * v[vb + 2] +
          r.gN[0] * v[nb] + r.gN[1] * v[nb + 1] + r.gN[2] * v[nb + 2] +
          r.gB * v[nb + 3] +
          r.gD * v[di];
      } else {
        const nb = this.nBase(r.fi);
        out[i] = r.gN[0] * v[nb] + r.gN[1] * v[nb + 1] + r.gN[2] * v[nb + 2];
      }
    }
    return out;
  }

  private jtTimesVector(rows: ReadonlyArray<LinearizedRow>, w: ReadonlyArray<number>): number[] {
    const out = new Array<number>(this.fullDim()).fill(0);
    for (let i = 0; i < rows.length; i++) {
      const wi = w[i];
      const r = rows[i];
      if (r.kind === "inc") {
        const vb = 3 * r.vi;
        const nb = this.nBase(r.fi);

        out[vb] += wi * r.gV[0];
        out[vb + 1] += wi * r.gV[1];
        out[vb + 2] += wi * r.gV[2];

        out[nb] += wi * r.gN[0];
        out[nb + 1] += wi * r.gN[1];
        out[nb + 2] += wi * r.gN[2];
        out[nb + 3] += wi * r.gB;
      } else if (r.kind === "noninc") {
        const vb = 3 * r.vi;
        const nb = this.nBase(r.fi);
        const di = this.dIndex(r.di);

        out[vb] += wi * r.gV[0];
        out[vb + 1] += wi * r.gV[1];
        out[vb + 2] += wi * r.gV[2];

        out[nb] += wi * r.gN[0];
        out[nb + 1] += wi * r.gN[1];
        out[nb + 2] += wi * r.gN[2];
        out[nb + 3] += wi * r.gB;

        out[di] += wi * r.gD;
      } else {
        const nb = this.nBase(r.fi);
        out[nb] += wi * r.gN[0];
        out[nb + 1] += wi * r.gN[1];
        out[nb + 2] += wi * r.gN[2];
      }
    }
    return out;
  }

  private objectiveAndGradient(y: ReadonlyArray<number>, gradOut?: number[]): number {
    const vDim = this.vertexDim();
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

    // Objective does not depend on plane/slack variables.
    if (gradOut && gradOut.length > vDim) {
      for (let i = vDim; i < gradOut.length; i++) gradOut[i] = 0;
    }
    return f;
  }

  private merit(
    y: ReadonlyArray<number>,
    rho: number,
    u: ReadonlyArray<number>,
    proxWeight: number,
    yRef: ReadonlyArray<number>,
    normalProxWeight: number,
    offsetProxWeight: number
  ): number {
    const f = this.objectiveAndGradient(y);
    const c = this.evalConstraintsOnly(y);
    let pen = 0;
    for (let i = 0; i < c.length; i++) {
      const t = c[i] + u[i];
      pen += t * t;
    }

    let proxV = 0;
    if (proxWeight > 0) {
      for (let i = 0; i < this.x.length; i++) {
        const b = 3 * i;
        const dx = y[b] - this.x0[i][0];
        const dy = y[b + 1] - this.x0[i][1];
        const dz = y[b + 2] - this.x0[i][2];
        proxV += dx * dx + dy * dy + dz * dz;
      }
    }
    let proxNB = 0;
    if (normalProxWeight > 0 || offsetProxWeight > 0) {
      for (let fi = 0; fi < this.faces.length; fi++) {
        const nb = this.nBase(fi);
        if (normalProxWeight > 0) {
          const dx = y[nb] - yRef[nb];
          const dy = y[nb + 1] - yRef[nb + 1];
          const dz = y[nb + 2] - yRef[nb + 2];
          proxNB += normalProxWeight * (dx * dx + dy * dy + dz * dz);
        }
        if (offsetProxWeight > 0) {
          const db = y[nb + 3] - yRef[nb + 3];
          proxNB += offsetProxWeight * db * db;
        }
      }
    }

    return f + 0.5 * rho * pen + 0.5 * proxWeight * proxV + 0.5 * proxNB;
  }

  private computeTotalPlanarityViolation(positions: ReadonlyArray<Vec3>): number {
    return sumSquaredPlanarityResidual(this.faces, positions);
  }

  step(iterations: number) {
    if (iterations <= 0) return;
    if (this.faces.length === 0 || this.x.length === 0) return;

    const dim = this.fullDim();

    const tau = Math.max(1e-10, this.params.tau ?? 1e-6);
    const proxWeight = Math.max(0, this.params.proxWeight ?? 0);
    const normalProxWeight = Math.max(0, this.params.normalProxWeight ?? 1);
    const offsetProxWeight = Math.max(0, this.params.offsetProxWeight ?? 1);
    const minAcceptedAlpha = Math.max(0, Math.min(1, this.params.minAcceptedAlpha ?? 1e-4));
    const cgIters = Math.max(4, Math.floor(this.params.cgIters ?? 80));
    const cgTol = Math.max(1e-10, this.params.cgTol ?? 1e-6);
    const lineSearchC1 = this.params.lineSearchC1 ?? 1e-4;
    const lineSearchShrink = Math.min(0.95, Math.max(0.1, this.params.lineSearchShrink ?? 0.5));
    const lineSearchMaxSteps = Math.max(1, Math.floor(this.params.lineSearchMaxSteps ?? 8));

    const adaptRho = this.params.adaptRho ?? false;
    const rhoInc = Math.max(1.01, this.params.rhoIncrease ?? 2);
    const rhoDec = Math.max(1.01, this.params.rhoDecrease ?? 2);
    const rhoRatio = Math.max(1.1, this.params.rhoResidualRatio ?? 10);
    const rhoMin = Math.max(1e-8, this.params.rhoMin ?? 1e-3);
    const rhoMax = Math.max(rhoMin, this.params.rhoMax ?? 1e8);

    let y = this.flattenY();
    const yRef = y.slice();
    let rho = Math.max(rhoMin, Math.min(rhoMax, this.params.rho));

    for (let out = 0; out < iterations; out++) {
      const { rows, c0 } = this.linearizeConstraints(y);
      const gradF = new Array<number>(dim).fill(0);
      this.objectiveAndGradient(y, gradF);

      const hDiag = new Array<number>(dim).fill(tau);
      for (let i = 0; i < this.x.length; i++) {
        const b = 3 * i;
        const isHandle = this.handles.targets.has(i);
        const w = isHandle ? this.params.wHandle : this.params.wFree;
        const val = 2 * w + proxWeight + tau;
        hDiag[b] = val;
        hDiag[b + 1] = val;
        hDiag[b + 2] = val;
      }
      for (let fi = 0; fi < this.faces.length; fi++) {
        const nb = this.nBase(fi);
        hDiag[nb] += normalProxWeight;
        hDiag[nb + 1] += normalProxWeight;
        hDiag[nb + 2] += normalProxWeight;
        hDiag[nb + 3] += offsetProxWeight;
      }

      const cPlusU = new Array<number>(c0.length);
      for (let i = 0; i < c0.length; i++) cPlusU[i] = c0[i] + this.u[i];

      const jtC = this.jtTimesVector(rows, cPlusU);

      const rhs = new Array<number>(dim).fill(0);
      for (let i = 0; i < dim; i++) rhs[i] = gradF[i] + rho * jtC[i];
      if (proxWeight > 0) {
        for (let i = 0; i < this.x.length; i++) {
          const b = 3 * i;
          rhs[b] += proxWeight * (y[b] - this.x0[i][0]);
          rhs[b + 1] += proxWeight * (y[b + 1] - this.x0[i][1]);
          rhs[b + 2] += proxWeight * (y[b + 2] - this.x0[i][2]);
        }
      }
      if (normalProxWeight > 0 || offsetProxWeight > 0) {
        for (let fi = 0; fi < this.faces.length; fi++) {
          const nb = this.nBase(fi);
          rhs[nb] += normalProxWeight * (y[nb] - yRef[nb]);
          rhs[nb + 1] += normalProxWeight * (y[nb + 1] - yRef[nb + 1]);
          rhs[nb + 2] += normalProxWeight * (y[nb + 2] - yRef[nb + 2]);
          rhs[nb + 3] += offsetProxWeight * (y[nb + 3] - yRef[nb + 3]);
        }
      }

      const applyA = (v: number[]): number[] => {
        const jv = this.jTimesVector(rows, v);
        const jtjv = this.jtTimesVector(rows, jv);
        const outA = new Array<number>(dim);
        for (let i = 0; i < dim; i++) outA[i] = hDiag[i] * v[i] + rho * jtjv[i];
        return outA;
      };

      const b = rhs.map((r) => -r);
      const delta = solveCG(applyA, b, cgIters, cgTol);
      const dirDeriv = dotN(rhs, delta);

      const psi0 = this.merit(y, rho, this.u, proxWeight, yRef, normalProxWeight, offsetProxWeight);
      let alpha = 1;
      let accepted = false;
      let yNext = y;

      if (dirDeriv < 0) {
        for (let ls = 0; ls < lineSearchMaxSteps; ls++) {
          const trial = new Array<number>(dim);
          for (let i = 0; i < dim; i++) trial[i] = y[i] + alpha * delta[i];
          const psiTrial = this.merit(trial, rho, this.u, proxWeight, yRef, normalProxWeight, offsetProxWeight);
          if (psiTrial <= psi0 + lineSearchC1 * alpha * dirDeriv) {
            accepted = true;
            yNext = trial;
            break;
          }
          alpha *= lineSearchShrink;
        }
      }

      if (!accepted || alpha < minAcceptedAlpha) break;
      y = yNext;

      const cNew = this.evalConstraintsOnly(y);
      for (let i = 0; i < this.u.length; i++) this.u[i] += cNew[i];

      if (adaptRho) {
        const primal = normN(cNew);
        const dc = new Array<number>(cNew.length);
        for (let i = 0; i < cNew.length; i++) dc[i] = cNew[i] - c0[i];
        const dual = rho * normN(this.jtTimesVector(rows, dc));

        let rhoNew = rho;
        if (primal > rhoRatio * dual) rhoNew = Math.min(rhoMax, rho * rhoInc);
        else if (dual > rhoRatio * primal) rhoNew = Math.max(rhoMin, rho / rhoDec);

        if (rhoNew !== rho) {
          const scale = rho / rhoNew;
          for (let i = 0; i < this.u.length; i++) this.u[i] *= scale;
          rho = rhoNew;
        }
      }
    }

    this.params.rho = rho;
    this.syncFromY(y);
    this.lastTotalViolation = this.computeTotalPlanarityViolation(this.x);
  }

  getPositionsRef(): ReadonlyArray<Vec3> {
    return this.x;
  }

  snapshotPositions(): Vec3[] {
    return this.x.map((p) => [p[0], p[1], p[2]] as Vec3);
  }

  diagnostics() {
    return { totalPlanarityViolation: this.lastTotalViolation };
  }
}
