import type { MetaModel, MetaState, PrimalEngine, StepProposal } from "../types";

function dotN(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function solveCG(
  applyA: (p: number[]) => number[],
  b: ReadonlyArray<number>,
  maxIters: number,
  tol: number
): number[] {
  const n = b.length;
  const x = new Array<number>(n).fill(0);
  let r = Array.from(b);
  let p = r.slice();
  let rr = dotN(r, r);
  if (Math.sqrt(rr) <= tol) return x;

  for (let it = 0; it < maxIters; it++) {
    const Ap = applyA(p);
    const pAp = Math.max(1e-20, dotN(p, Ap));
    const alpha = rr / pAp;
    for (let i = 0; i < n; i++) x[i] += alpha * p[i];
    for (let i = 0; i < n; i++) r[i] -= alpha * Ap[i];
    const rrNew = dotN(r, r);
    if (Math.sqrt(rrNew) <= tol) break;
    const beta = rrNew / Math.max(1e-20, rr);
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    rr = rrNew;
  }
  return x;
}

export type LinearizedAlmEngineParams = {
  cgIters: number;
  cgTol: number;
};

export class LinearizedAlmEngine implements PrimalEngine {
  private params: LinearizedAlmEngineParams;

  constructor(params: LinearizedAlmEngineParams) {
    this.params = params;
  }

  setParams(next: Partial<LinearizedAlmEngineParams>) {
    this.params = { ...this.params, ...next };
  }

  propose(state: Readonly<MetaState>, model: Readonly<MetaModel>): StepProposal {
    const c0 = model.hard.linearization.c0;
    const cPlusU = new Array<number>(c0.length);
    for (let i = 0; i < c0.length; i++) cPlusU[i] = c0[i] + state.u[i];

    const jtC = model.hard.linearization.applyJT(cPlusU);

    const rhs = new Array<number>(model.dim);
    for (let i = 0; i < model.dim; i++) rhs[i] = model.gradient[i] + state.rho * jtC[i];

    const applyA = (v: number[]): number[] => {
      const jv = model.hard.linearization.applyJ(v);
      const jtjv = model.hard.linearization.applyJT(jv);
      const out = new Array<number>(model.dim);
      for (let i = 0; i < model.dim; i++) out[i] = model.hDiag[i] * v[i] + state.rho * jtjv[i];
      return out;
    };

    const b = rhs.map((r) => -r);
    const delta = solveCG(applyA, b, this.params.cgIters, this.params.cgTol);
    const directionalDerivative = dotN(rhs, delta);
    return { delta, directionalDerivative };
  }
}

