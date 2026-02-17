export type ConstraintSense = "eq" | "le";

export type DenseMatrix = number[][];

export type QuadraticForm = {
  A: DenseMatrix;
  b: number[];
  c: number;
};

export interface ScalarFunction {
  evaluate: (x: ReadonlyArray<number>) => number;
  jacobian: (x: ReadonlyArray<number>) => number[];
  hessian: (x: ReadonlyArray<number>) => DenseMatrix;
}

export type FunctionConstraint = {
  id: string;
  sense: ConstraintSense;
  fn: ScalarFunction;
};

export type QuadraticConstraint = {
  id: string;
  sense: ConstraintSense;
  form: QuadraticForm;
};

export type QuadraticConstraintSet = {
  equalities: QuadraticConstraint[];
  inequalities: QuadraticConstraint[];
};

export type FunctionConstraintSet = {
  equalities: FunctionConstraint[];
  inequalities: FunctionConstraint[];
};

export type QuadraticProvider = (x: ReadonlyArray<number>) => QuadraticConstraintSet;

export type QuadraticObjectiveProvider = (x: ReadonlyArray<number>) => QuadraticForm;

export type OptimizationModel = {
  quadraticConstraints: QuadraticConstraintSet;
  exactConstraints: FunctionConstraintSet;
  localQuadraticConstraints?: QuadraticProvider;
  metric?: ScalarFunction;
  regularizer?: ScalarFunction;
  localQuadraticMetric?: QuadraticObjectiveProvider;
  localQuadraticRegularizer?: QuadraticObjectiveProvider;
};

export interface OptimizerKernel<State, Params, Memory, Model = OptimizationModel> {
  initialize: (args: {
    model: Model;
    state: State;
    params: Params;
    resume?: Memory;
  }) => Memory;
  step: (args: {
    model: Model;
    state: State;
    params: Params;
    memory: Memory;
    iterations: number;
  }) => void;
}
