import type {
  DualUpdater,
  Globalizer,
  MetaModel,
  MetaState,
  PenaltyPolicy,
  StepAcceptance,
  StepProposal,
  StopPolicy,
} from "./types";

function dotN(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function normN(a: ReadonlyArray<number>): number {
  return Math.sqrt(Math.max(0, dotN(a, a)));
}

export type ArmijoParams = {
  c1: number;
  shrink: number;
  maxSteps: number;
};

export function createArmijoGlobalizer(params: ArmijoParams): Globalizer {
  const c1 = params.c1;
  const shrink = params.shrink;
  const maxSteps = params.maxSteps;

  const reject: StepAcceptance = { accepted: false, alpha: 0, yNext: [] };
  return {
    accept(state: Readonly<MetaState>, model: Readonly<MetaModel>, proposal: Readonly<StepProposal>): StepAcceptance {
      const dirDeriv = proposal.directionalDerivative;
      if (dirDeriv >= 0) return reject;

      const psi0 = model.merit(state.y, state.u, state.rho);
      let alpha = 1;

      for (let ls = 0; ls < maxSteps; ls++) {
        const trial = new Array<number>(proposal.delta.length);
        for (let i = 0; i < trial.length; i++) trial[i] = state.y[i] + alpha * proposal.delta[i];
        const psiTrial = model.merit(trial, state.u, state.rho);
        if (psiTrial <= psi0 + c1 * alpha * dirDeriv) {
          return { accepted: true, alpha, yNext: trial };
        }
        alpha *= shrink;
      }

      return reject;
    },
  };
}

export const scaledDualUpdater: DualUpdater = {
  update(state: MetaState, cNew: ReadonlyArray<number>) {
    for (let i = 0; i < state.u.length; i++) state.u[i] += cNew[i];
  },
};

export type ResidualBalancePenaltyParams = {
  enabled: boolean;
  increase: number;
  decrease: number;
  ratio: number;
  min: number;
  max: number;
};

export function createResidualBalancePenaltyPolicy(params: ResidualBalancePenaltyParams): PenaltyPolicy {
  return {
    update(state: MetaState, model: Readonly<MetaModel>, cNew: ReadonlyArray<number>) {
      if (!params.enabled) return;

      const cPrev = model.hard.linearization.c0;
      const dc = new Array<number>(cNew.length);
      for (let i = 0; i < cNew.length; i++) dc[i] = cNew[i] - cPrev[i];

      const primal = normN(cNew);
      const dual = state.rho * normN(model.hard.linearization.applyJT(dc));

      let rhoNew = state.rho;
      if (primal > params.ratio * dual) rhoNew = Math.min(params.max, state.rho * params.increase);
      else if (dual > params.ratio * primal) rhoNew = Math.max(params.min, state.rho / params.decrease);

      if (rhoNew === state.rho) return;
      const scale = state.rho / rhoNew;
      for (let i = 0; i < state.u.length; i++) state.u[i] *= scale;
      state.rho = rhoNew;
    },
  };
}

export const neverStopPolicy: StopPolicy = {
  shouldStop() {
    return false;
  },
};

export type ResidualStopParams = {
  constraintTol: number;
};

export function createResidualStopPolicy(params: ResidualStopParams): StopPolicy {
  return {
    shouldStop(_state: Readonly<MetaState>, _model: Readonly<MetaModel>, cNew: ReadonlyArray<number>) {
      return normN(cNew) <= params.constraintTol;
    },
  };
}

