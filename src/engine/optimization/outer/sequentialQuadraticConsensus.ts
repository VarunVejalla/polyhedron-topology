import { quadraticizeAt, zeroQuadratic } from "../quadratic";
import { OptimizerSession } from "../session";
import type {
  ConstraintSense,
  IndexedQuadraticConstraint,
  IndexedQuadraticConstraintSet,
  IndexedQuadraticForm,
  OptimizationModel,
  QuadraticConstraint,
  QuadraticConstraintSet,
  QuadraticForm,
} from "../types";
import {
  consensusQcqpKernel,
  type ConsensusConstraint,
  type ConsensusQcqpMemory,
  type ConsensusQcqpModel,
  type ConsensusQcqpParams,
  type ConsensusQcqpState,
} from "../kernels/consensusQcqpKernel";

export type SequentialConsensusParams = {
  rho: number;
  proximalWeight: number;
  linearSolveShift: number;
  qcqpTol: number;
  qcqpMaxNewtonIters: number;
  relinearizeEvery: number;
  innerIterationsPerOuter: number;
};

function mergeQuadratic(a: Readonly<QuadraticForm>, b: Readonly<QuadraticForm>): QuadraticForm {
  const dim = a.b.length;
  const A = Array.from({ length: dim }, () => new Array<number>(dim).fill(0));
  const bb = new Array<number>(dim).fill(0);
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) A[i][j] = (a.A[i]?.[j] ?? 0) + (b.A[i]?.[j] ?? 0);
    bb[i] = (a.b[i] ?? 0) + (b.b[i] ?? 0);
  }
  return { A, b: bb, c: a.c + b.c };
}

function supportFromDense(form: Readonly<QuadraticForm>, eps = 1e-12): number[] {
  const n = form.b.length;
  const idx: number[] = [];
  for (let i = 0; i < n; i++) {
    let keep = Math.abs(form.b[i] ?? 0) > eps;
    if (!keep) {
      for (let j = 0; j < n; j++) {
        if (Math.abs(form.A[i]?.[j] ?? 0) > eps || Math.abs(form.A[j]?.[i] ?? 0) > eps) {
          keep = true;
          break;
        }
      }
    }
    if (keep) idx.push(i);
  }
  return idx;
}

function denseToIndexedForm(form: Readonly<QuadraticForm>, eps = 1e-12): IndexedQuadraticForm {
  const indices = supportFromDense(form, eps);
  const k = indices.length;
  const A = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const b = new Array<number>(k).fill(0);
  for (let i = 0; i < k; i++) {
    const gi = indices[i];
    b[i] = form.b[gi] ?? 0;
    for (let j = 0; j < k; j++) A[i][j] = form.A[gi]?.[indices[j]] ?? 0;
  }
  return { indices, A, b, c: form.c };
}

function collectDenseConstraints(set: QuadraticConstraintSet, prefix: string): ConsensusConstraint[] {
  const out: ConsensusConstraint[] = [];
  const push = (list: QuadraticConstraint[], sense: ConstraintSense, tag: string) => {
    for (let i = 0; i < list.length; i++) {
      out.push({
        id: `${prefix}:${tag}:${list[i].id}`,
        sense,
        form: denseToIndexedForm(list[i].form),
      });
    }
  };
  push(set.equalities, "eq", "eq");
  push(set.inequalities, "le", "le");
  return out;
}

function collectIndexedConstraints(set: IndexedQuadraticConstraintSet, prefix: string): ConsensusConstraint[] {
  const out: ConsensusConstraint[] = [];
  const push = (list: IndexedQuadraticConstraint[], sense: ConstraintSense, tag: string) => {
    for (let i = 0; i < list.length; i++) {
      out.push({
        id: `${prefix}:${tag}:${list[i].id}`,
        sense,
        form: {
          indices: list[i].form.indices.slice(),
          A: list[i].form.A.map((r) => r.slice()),
          b: list[i].form.b.slice(),
          c: list[i].form.c,
        },
      });
    }
  };
  push(set.equalities, "eq", "eq");
  push(set.inequalities, "le", "le");
  return out;
}

function collectFunctionQuadraticConstraints(model: OptimizationModel, xRef: ReadonlyArray<number>): ConsensusConstraint[] {
  const out: ConsensusConstraint[] = [];
  for (let i = 0; i < model.exactConstraints.equalities.length; i++) {
    const c = model.exactConstraints.equalities[i];
    out.push({
      id: `fx:eq:${c.id}`,
      sense: "eq",
      form: denseToIndexedForm(quadraticizeAt(c.fn, xRef)),
    });
  }
  for (let i = 0; i < model.exactConstraints.inequalities.length; i++) {
    const c = model.exactConstraints.inequalities[i];
    out.push({
      id: `fx:le:${c.id}`,
      sense: "le",
      form: denseToIndexedForm(quadraticizeAt(c.fn, xRef)),
    });
  }
  return out;
}

function buildObjective(model: OptimizationModel, xRef: ReadonlyArray<number>): QuadraticForm {
  const dim = xRef.length;
  const zero = zeroQuadratic(dim);
  const metric = model.localQuadraticMetric
    ? model.localQuadraticMetric(xRef)
    : model.metric
      ? quadraticizeAt(model.metric, xRef)
      : zero;
  const regularizer = model.localQuadraticRegularizer
    ? model.localQuadraticRegularizer(xRef)
    : model.regularizer
      ? quadraticizeAt(model.regularizer, xRef)
      : zero;
  return mergeQuadratic(metric, regularizer);
}

function buildConsensusSurrogateModel(model: OptimizationModel, xRef: ReadonlyArray<number>): ConsensusQcqpModel {
  const constraints: ConsensusConstraint[] = [];
  constraints.push(...collectDenseConstraints(model.quadraticConstraints, "q"));
  if (model.indexedQuadraticConstraints) constraints.push(...collectIndexedConstraints(model.indexedQuadraticConstraints, "iq"));
  if (model.localQuadraticConstraints) constraints.push(...collectDenseConstraints(model.localQuadraticConstraints(xRef), "lq"));
  constraints.push(...collectFunctionQuadraticConstraints(model, xRef));

  return {
    dim: xRef.length,
    objective: buildObjective(model, xRef),
    constraints,
    sourceModel: model,
  };
}

export class SequentialQuadraticConsensusSolver {
  private baseModel: OptimizationModel;
  private params: SequentialConsensusParams;
  private session: OptimizerSession<ConsensusQcqpState, ConsensusQcqpParams, ConsensusQcqpMemory, ConsensusQcqpModel>;
  private stepsSinceRelinearize = 0;

  constructor(baseModel: OptimizationModel, x0: number[], params: SequentialConsensusParams) {
    this.baseModel = baseModel;
    this.params = { ...params };
    this.session = new OptimizerSession({
      kernel: consensusQcqpKernel,
      model: buildConsensusSurrogateModel(this.baseModel, x0),
      initialState: { x: x0.slice() },
      params: this.kernelParams(),
    });
  }

  private kernelParams(): ConsensusQcqpParams {
    return {
      rho: Math.max(1e-8, this.params.rho),
      proximalWeight: Math.max(0, this.params.proximalWeight),
      linearSolveShift: Math.max(0, this.params.linearSolveShift),
      qcqpTol: Math.max(1e-12, this.params.qcqpTol),
      qcqpMaxNewtonIters: Math.max(4, Math.floor(this.params.qcqpMaxNewtonIters)),
    };
  }

  setModel(next: OptimizationModel): void {
    this.baseModel = next;
    this.stepsSinceRelinearize = Math.max(1, this.params.relinearizeEvery);
  }

  setState(x: number[]): void {
    this.session.setState({ x: x.slice() });
    this.stepsSinceRelinearize = Math.max(1, this.params.relinearizeEvery);
  }

  getStateRef(): Readonly<ConsensusQcqpState> {
    return this.session.getStateRef();
  }

  getMemoryRef(): Readonly<ConsensusQcqpMemory> {
    return this.session.getMemoryRef();
  }

  setParams(next: Partial<SequentialConsensusParams>): void {
    this.params = { ...this.params, ...next };
    this.session.setParams(this.kernelParams());
  }

  step(iterations: number): void {
    const n = Math.max(0, Math.floor(iterations));
    if (n <= 0) return;
    const relinearizeEvery = Math.max(1, Math.floor(this.params.relinearizeEvery));
    const innerPerOuter = Math.max(1, Math.floor(this.params.innerIterationsPerOuter));

    for (let i = 0; i < n; i++) {
      if (this.stepsSinceRelinearize >= relinearizeEvery) {
        this.session.setModel(buildConsensusSurrogateModel(this.baseModel, this.session.getStateRef().x));
        this.stepsSinceRelinearize = 0;
      }
      this.session.step(innerPerOuter);
      this.stepsSinceRelinearize += 1;
    }
  }
}
