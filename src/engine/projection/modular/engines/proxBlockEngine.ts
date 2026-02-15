import type { MetaModel, MetaState, PrimalEngine, StepProposal } from "../types";

// Placeholder engine for future CAD2015-style prox block updates.
// It currently proposes no movement; keep this as a swappable slot in the modular framework.
export class ProxBlockEngine implements PrimalEngine {
  propose(_state: Readonly<MetaState>, model: Readonly<MetaModel>): StepProposal {
    return {
      delta: new Array<number>(model.dim).fill(0),
      directionalDerivative: 0,
    };
  }
}

