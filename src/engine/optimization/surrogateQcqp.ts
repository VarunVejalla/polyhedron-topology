import { quadraticizeAt, zeroQuadratic } from "./quadratic";
import type {
  ConstraintSense,
  IndexedQuadraticConstraint,
  IndexedQuadraticConstraintSet,
  IndexedQuadraticForm,
  OptimizationModel,
  QuadraticConstraint,
  QuadraticConstraintSet,
  QuadraticForm,
} from "./types";

type SurrogateQcqpConstraint = {
  id: string;
  sense: ConstraintSense;
  form: IndexedQuadraticForm;
};

export type SurrogateQcqpModel = {
  dim: number;
  objective: QuadraticForm;
  constraints: SurrogateQcqpConstraint[];
  sourceModel?: OptimizationModel;
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

function collectDenseConstraints(set: QuadraticConstraintSet, prefix: string): SurrogateQcqpConstraint[] {
  const out: SurrogateQcqpConstraint[] = [];
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

function collectIndexedConstraints(set: IndexedQuadraticConstraintSet, prefix: string): SurrogateQcqpConstraint[] {
  const out: SurrogateQcqpConstraint[] = [];
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

function collectFunctionQuadraticConstraints(model: OptimizationModel, xRef: ReadonlyArray<number>): SurrogateQcqpConstraint[] {
  const out: SurrogateQcqpConstraint[] = [];
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

export function buildQuadraticSurrogateModel(model: OptimizationModel, xRef: ReadonlyArray<number>): SurrogateQcqpModel {
  const constraints: SurrogateQcqpConstraint[] = [];
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
