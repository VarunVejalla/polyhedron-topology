type ConstraintSense = "eq" | "le";

export type SymmetricEntry = {
  i: number;
  j: number;
  value: number;
};

export interface SymmetricOperator {
  dim: number;
  apply: (v: ReadonlyArray<number>) => number[];
  entries?: ReadonlyArray<SymmetricEntry>;
}

export type QuadraticForm = {
  dim: number;
  A: SymmetricOperator;
  b: number[];
  c: number;
};

export type QuadraticConstraint = {
  id: string;
  sense: ConstraintSense;
  form: QuadraticForm;
  source: "exact" | "local";
};

export type OptimizationProblem = {
  dim: number;
  xRef: number[];
  exactEq: QuadraticConstraint[];
  exactLe: QuadraticConstraint[];
  localEq: QuadraticConstraint[];
  localLe: QuadraticConstraint[];
  metric?: QuadraticForm;
  regularizer?: QuadraticForm;
  objectiveValueOverride?: (x: ReadonlyArray<number>) => number;
};

export type OptimizerHyperParams = {
  rho: number;
  tau?: number;
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
};

export type OptimizerState = {
  x: number[];
  rho: number;
  eqDual: number[];
  leDual: number[];
  eqIds: string[];
  leIds: string[];
};

export type StepReport = {
  attempted: number;
  accepted: number;
  lastAlpha: number;
  eqResidualL2: number;
  leViolationMax: number;
  objective: number;
};

export interface OptimizerKernel {
  id: string;
  initialize: (
    problem: Readonly<OptimizationProblem>,
    x0: ReadonlyArray<number>,
    hp: Readonly<OptimizerHyperParams>
  ) => OptimizerState;
  rebindProblem: (
    problem: Readonly<OptimizationProblem>,
    state: OptimizerState,
    hp: Readonly<OptimizerHyperParams>
  ) => void;
  step: (
    problem: Readonly<OptimizationProblem>,
    state: OptimizerState,
    hp: Readonly<OptimizerHyperParams>,
    iterations: number
  ) => StepReport;
}

export interface ProblemBuilder<Ctx = unknown> {
  initializeContext?: (x0: ReadonlyArray<number>, hp: Readonly<OptimizerHyperParams>) => Ctx;
  updateContext?: (x: ReadonlyArray<number>, ctx: Ctx, hp: Readonly<OptimizerHyperParams>) => Ctx;
  buildProblem: (
    xRef: ReadonlyArray<number>,
    ctx: Ctx,
    hp: Readonly<OptimizerHyperParams>
  ) => OptimizationProblem;
}

export interface PieceSelector<Ctx = unknown> {
  keyAt: (x: ReadonlyArray<number>, ctx: Ctx, hp: Readonly<OptimizerHyperParams>) => string;
}
