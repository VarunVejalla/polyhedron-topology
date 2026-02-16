import type { MetaModel, MetaState, PrimalEngine, StepProposal } from "../types";
import { dotN, solveCG } from "../../shared/numeric";

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

