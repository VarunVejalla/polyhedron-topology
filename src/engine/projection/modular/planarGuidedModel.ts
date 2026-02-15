import type { Vec3 } from "../../math/types";
import { bestFitPlanePCA } from "../../geom/plane";
import type { HandleSet } from "../index";
import type { ConstraintLinearization, MetaModel, MetaModelBuilder, MetaState } from "./types";

type Incidence = {
  fi: number;
  vi: number;
};

type IncLinearized = {
  kind: "inc";
  fi: number;
  vi: number;
  gV: Vec3;
  gN: Vec3;
  gB: number;
};

type UnitLinearized = {
  kind: "unit";
  fi: number;
  gN: Vec3;
};

type LinearizedRow = IncLinearized | UnitLinearized;

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
};

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
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

export function countPlanarGuidedHardConstraints(faces: ReadonlyArray<ReadonlyArray<number>>): number {
  let m = faces.length;
  for (let fi = 0; fi < faces.length; fi++) m += faces[fi].length;
  return m;
}

export function packPlanarGuidedY(vertices: ReadonlyArray<Vec3>, normals: ReadonlyArray<Vec3>, offsets: ReadonlyArray<number>): number[] {
  const out = new Array<number>(3 * vertices.length + 4 * normals.length);
  for (let i = 0; i < vertices.length; i++) {
    const b = 3 * i;
    out[b] = vertices[i][0];
    out[b + 1] = vertices[i][1];
    out[b + 2] = vertices[i][2];
  }
  const nStart = 3 * vertices.length;
  for (let fi = 0; fi < normals.length; fi++) {
    const b = nStart + 4 * fi;
    out[b] = normals[fi][0];
    out[b + 1] = normals[fi][1];
    out[b + 2] = normals[fi][2];
    out[b + 3] = offsets[fi];
  }
  return out;
}

export function unpackPlanarGuidedY(y: ReadonlyArray<number>, vertexCount: number, faceCount: number): { vertices: Vec3[]; normals: Vec3[]; offsets: number[] } {
  const vertices: Vec3[] = new Array(vertexCount);
  const normals: Vec3[] = new Array(faceCount);
  const offsets: number[] = new Array(faceCount);
  for (let i = 0; i < vertexCount; i++) {
    const b = 3 * i;
    vertices[i] = [y[b], y[b + 1], y[b + 2]];
  }
  const nStart = 3 * vertexCount;
  for (let fi = 0; fi < faceCount; fi++) {
    const b = nStart + 4 * fi;
    normals[fi] = [y[b], y[b + 1], y[b + 2]];
    offsets[fi] = y[b + 3];
  }
  return { vertices, normals, offsets };
}

export function computeTotalPlanarityViolation(
  faces: ReadonlyArray<ReadonlyArray<number>>,
  positions: ReadonlyArray<Vec3>
): number {
  let total = 0;
  for (let fi = 0; fi < faces.length; fi++) {
    const face = faces[fi];
    if (face.length < 3) continue;
    const pts = face.map((vi) => positions[vi]);
    const plane = bestFitPlanePCA(pts);
    const n = plane.n;
    const b = plane.b;
    for (let k = 0; k < face.length; k++) {
      const p = positions[face[k]];
      total += Math.abs(n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - b);
    }
  }
  return total;
}

export class PlanarGuidedModelBuilder implements MetaModelBuilder {
  private faces: number[][];
  private x0: Vec3[];
  private params: ModularGuidedALMParams;
  private handles: HandleSet;
  private yRefProvider: () => ReadonlyArray<number>;
  private incidences: Incidence[] = [];

  constructor(
    faces: number[][],
    x0: Vec3[],
    params: ModularGuidedALMParams,
    handles: HandleSet,
    yRefProvider: () => ReadonlyArray<number>
  ) {
    this.faces = faces.map((f) => [...f]);
    this.x0 = x0.map((p) => [p[0], p[1], p[2]]);
    this.params = { ...params };
    this.handles = handles;
    this.yRefProvider = yRefProvider;
    this.buildIncidences();
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
    return this.incidences.length + this.faces.length;
  }

  private buildIncidences() {
    this.incidences = [];
    for (let fi = 0; fi < this.faces.length; fi++) {
      for (const vi of this.faces[fi]) this.incidences.push({ fi, vi });
    }
  }

  private vertexDim(): number {
    return 3 * this.x0.length;
  }

  private fullDim(): number {
    return this.vertexDim() + 4 * this.faces.length;
  }

  private nBase(fi: number): number {
    return this.vertexDim() + 4 * fi;
  }

  private bIndex(fi: number): number {
    return this.vertexDim() + 4 * fi + 3;
  }

  private linearizeConstraints(y: ReadonlyArray<number>): ConstraintLinearization {
    const mInc = this.incidences.length;
    const rows: LinearizedRow[] = new Array(mInc + this.faces.length);
    const c0 = new Array<number>(rows.length);

    for (let ri = 0; ri < mInc; ri++) {
      const { fi, vi } = this.incidences[ri];
      const vb = 3 * vi;
      const nb = this.nBase(fi);
      const v: Vec3 = [y[vb], y[vb + 1], y[vb + 2]];
      const n: Vec3 = [y[nb], y[nb + 1], y[nb + 2]];
      const b = y[this.bIndex(fi)];
      rows[ri] = {
        kind: "inc",
        fi,
        vi,
        gV: [n[0], n[1], n[2]],
        gN: [v[0], v[1], v[2]],
        gB: -1,
      };
      c0[ri] = dot3(n, v) - b;
    }

    for (let fi = 0; fi < this.faces.length; fi++) {
      const idx = mInc + fi;
      const nb = this.nBase(fi);
      const n: Vec3 = [y[nb], y[nb + 1], y[nb + 2]];
      rows[idx] = { kind: "unit", fi, gN: [2 * n[0], 2 * n[1], 2 * n[2]] };
      c0[idx] = dot3(n, n) - 1;
    }

    const applyJ = (v: ReadonlyArray<number>): number[] => {
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
        } else {
          const nb = this.nBase(r.fi);
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
          const nb = this.nBase(r.fi);
          out[vb] += wi * r.gV[0];
          out[vb + 1] += wi * r.gV[1];
          out[vb + 2] += wi * r.gV[2];
          out[nb] += wi * r.gN[0];
          out[nb + 1] += wi * r.gN[1];
          out[nb + 2] += wi * r.gN[2];
          out[nb + 3] += wi * r.gB;
        } else {
          const nb = this.nBase(r.fi);
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
    const out = new Array<number>(mInc + this.faces.length);
    for (let ri = 0; ri < mInc; ri++) {
      const { fi, vi } = this.incidences[ri];
      const vb = 3 * vi;
      const nb = this.nBase(fi);
      const v: Vec3 = [y[vb], y[vb + 1], y[vb + 2]];
      const n: Vec3 = [y[nb], y[nb + 1], y[nb + 2]];
      out[ri] = dot3(n, v) - y[this.bIndex(fi)];
    }
    for (let fi = 0; fi < this.faces.length; fi++) {
      const nb = this.nBase(fi);
      const n: Vec3 = [y[nb], y[nb + 1], y[nb + 2]];
      out[mInc + fi] = dot3(n, n) - 1;
    }
    return out;
  }

  private regularityValueAndGradient(positions: ReadonlyArray<Vec3>, epsArea: number, gradAccum?: Vec3[]): number {
    let total = 0;
    for (let fi = 0; fi < this.faces.length; fi++) {
      const face = this.faces[fi];
      const nSides = face.length;
      if (nSides < 3) continue;

      let perimeter = 0;
      for (let i = 0; i < nSides; i++) {
        const ia = face[i];
        const ib = face[(i + 1) % nSides];
        const ex = positions[ib][0] - positions[ia][0];
        const ey = positions[ib][1] - positions[ia][1];
        const ez = positions[ib][2] - positions[ia][2];
        perimeter += Math.hypot(ex, ey, ez);
      }

      const areaVec: Vec3 = [0, 0, 0];
      for (let i = 0; i < nSides; i++) {
        const ia = face[i];
        const ib = face[(i + 1) % nSides];
        const a = positions[ia];
        const b = positions[ib];
        areaVec[0] += 0.5 * (a[1] * b[2] - a[2] * b[1]);
        areaVec[1] += 0.5 * (a[2] * b[0] - a[0] * b[2]);
        areaVec[2] += 0.5 * (a[0] * b[1] - a[1] * b[0]);
      }

      const area = Math.sqrt(dot3(areaVec, areaVec) + epsArea * epsArea);
      const invArea = 1 / Math.max(1e-12, area);
      const nHat: Vec3 = [areaVec[0] * invArea, areaVec[1] * invArea, areaVec[2] * invArea];

      const cReg = 4 * nSides * Math.tan(Math.PI / nSides);
      const invDen = 1 / Math.max(1e-12, cReg * area);
      const reg = perimeter * perimeter * invDen - 1;
      const regSq = reg * reg;
      total += regSq;

      if (!gradAccum) continue;

      const regScale = 2 * reg;
      const coeffP = regScale * 2 * perimeter * invDen;
      const coeffA = regScale * (-(perimeter * perimeter) / Math.max(1e-12, cReg * area * area));
      for (let i = 0; i < nSides; i++) {
        const vi = face[i];
        const prev = face[(i - 1 + nSides) % nSides];
        const next = face[(i + 1) % nSides];

        const p = positions[vi];
        const pPrev = positions[prev];
        const pNext = positions[next];

        const ePrev: Vec3 = [p[0] - pPrev[0], p[1] - pPrev[1], p[2] - pPrev[2]];
        const eNext: Vec3 = [p[0] - pNext[0], p[1] - pNext[1], p[2] - pNext[2]];
        const lPrev = Math.max(1e-12, Math.hypot(ePrev[0], ePrev[1], ePrev[2]));
        const lNext = Math.max(1e-12, Math.hypot(eNext[0], eNext[1], eNext[2]));
        const dP: Vec3 = [
          ePrev[0] / lPrev + eNext[0] / lNext,
          ePrev[1] / lPrev + eNext[1] / lNext,
          ePrev[2] / lPrev + eNext[2] / lNext,
        ];

        const edgePN: Vec3 = [pNext[0] - pPrev[0], pNext[1] - pPrev[1], pNext[2] - pPrev[2]];
        const dA = cross3(edgePN, nHat);
        dA[0] *= 0.5;
        dA[1] *= 0.5;
        dA[2] *= 0.5;

        gradAccum[vi][0] += coeffP * dP[0] + coeffA * dA[0];
        gradAccum[vi][1] += coeffP * dP[1] + coeffA * dA[1];
        gradAccum[vi][2] += coeffP * dP[2] + coeffA * dA[2];
      }
    }
    return total;
  }

  private objectiveAndGradient(y: ReadonlyArray<number>, gradOut?: number[]): number {
    const epsArea = Math.max(1e-12, this.params.epsArea ?? 1e-8);
    const lambdaReg = Math.max(0, this.params.lambdaReg);
    if (gradOut) gradOut.fill(0);
    let f = 0;

    const positions: Vec3[] = new Array(this.x0.length);
    for (let i = 0; i < this.x0.length; i++) {
      const b = 3 * i;
      positions[i] = [y[b], y[b + 1], y[b + 2]];
    }

    for (let i = 0; i < this.x0.length; i++) {
      const b = 3 * i;
      const isHandle = this.handles.targets.has(i);
      const w = isHandle ? this.params.wHandle : this.params.wFree;
      const d = isHandle ? (this.handles.targets.get(i) as Vec3) : this.x0[i];
      const dx = y[b] - d[0];
      const dy = y[b + 1] - d[1];
      const dz = y[b + 2] - d[2];
      f += w * (dx * dx + dy * dy + dz * dz);
      if (gradOut) {
        gradOut[b] += 2 * w * dx;
        gradOut[b + 1] += 2 * w * dy;
        gradOut[b + 2] += 2 * w * dz;
      }
    }

    if (lambdaReg > 0) {
      const gradReg = gradOut ? positions.map(() => [0, 0, 0] as Vec3) : undefined;
      const reg = this.regularityValueAndGradient(positions, epsArea, gradReg);
      f += lambdaReg * reg;
      if (gradOut && gradReg) {
        for (let i = 0; i < gradReg.length; i++) {
          const b = 3 * i;
          gradOut[b] += lambdaReg * gradReg[i][0];
          gradOut[b + 1] += lambdaReg * gradReg[i][1];
          gradOut[b + 2] += lambdaReg * gradReg[i][2];
        }
      }
    }

    return f;
  }

  build(state: Readonly<MetaState>): MetaModel {
    const y = state.y;
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
        const nb = this.nBase(fi);
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
      const nb = this.nBase(fi);
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
            const nb = this.nBase(fi);
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

