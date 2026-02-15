export type ConstraintLinearization = {
  // c(y + delta) ~= c0 + J * delta
  c0: number[];
  applyJ: (v: ReadonlyArray<number>) => number[];
  applyJT: (w: ReadonlyArray<number>) => number[];
};

export type HardConstraintModel = {
  linearization: ConstraintLinearization;
  evaluate: (y: ReadonlyArray<number>) => number[];
};

export type MetaModel = {
  dim: number;
  gradient: number[];
  hDiag: number[];
  hard: HardConstraintModel;
  merit: (y: ReadonlyArray<number>, u: ReadonlyArray<number>, rho: number) => number;
};

export type MetaState = {
  y: number[];
  u: number[];
  rho: number;
};

export type StepProposal = {
  delta: number[];
  directionalDerivative: number;
};

export type StepAcceptance = {
  accepted: boolean;
  alpha: number;
  yNext: number[];
};

export type MetaStats = {
  attempted: number;
  accepted: number;
  lastAlpha: number;
};

export interface MetaModelBuilder {
  build: (state: Readonly<MetaState>) => MetaModel;
}

export interface PrimalEngine {
  propose: (state: Readonly<MetaState>, model: Readonly<MetaModel>) => StepProposal;
}

export interface Globalizer {
  accept: (state: Readonly<MetaState>, model: Readonly<MetaModel>, proposal: Readonly<StepProposal>) => StepAcceptance;
}

export interface DualUpdater {
  update: (state: MetaState, cNew: ReadonlyArray<number>) => void;
}

export interface PenaltyPolicy {
  update: (state: MetaState, model: Readonly<MetaModel>, cNew: ReadonlyArray<number>) => void;
}

export interface StopPolicy {
  shouldStop: (
    state: Readonly<MetaState>,
    model: Readonly<MetaModel>,
    cNew: ReadonlyArray<number>,
    stats: Readonly<MetaStats>
  ) => boolean;
}

