import { dotN } from "../projection/shared/numeric";
import type {
  QuadraticConstraint,
  QuadraticForm,
  SymmetricEntry,
  SymmetricOperator,
} from "./types";

function zeroOperator(dim: number): SymmetricOperator {
  return {
    dim,
    apply: () => new Array<number>(dim).fill(0),
    entries: [],
  };
}

function diagonalOperator(diag: ReadonlyArray<number>): SymmetricOperator {
  const entries: SymmetricEntry[] = new Array(diag.length);
  for (let i = 0; i < diag.length; i++) entries[i] = { i, j: i, value: diag[i] };
  return sparseSymmetricOperator(diag.length, entries);
}

export function sparseSymmetricOperator(dim: number, entries: ReadonlyArray<SymmetricEntry>): SymmetricOperator {
  const data = entries.map((e) => ({
    i: Math.min(e.i, e.j),
    j: Math.max(e.i, e.j),
    value: e.value,
  }));
  return {
    dim,
    entries: data,
    apply: (v: ReadonlyArray<number>) => {
      const out = new Array<number>(dim).fill(0);
      for (let k = 0; k < data.length; k++) {
        const e = data[k];
        if (e.i === e.j) {
          out[e.i] += e.value * v[e.i];
          continue;
        }
        out[e.i] += e.value * v[e.j];
        out[e.j] += e.value * v[e.i];
      }
      return out;
    },
  };
}

export function evaluateQuadraticForm(form: Readonly<QuadraticForm>, x: ReadonlyArray<number>): number {
  const entries = form.A.entries;
  let q = 0;
  if (entries && entries.length > 0) {
    for (let k = 0; k < entries.length; k++) {
      const e = entries[k];
      if (e.i === e.j) q += 0.5 * e.value * x[e.i] * x[e.i];
      else q += e.value * x[e.i] * x[e.j];
    }
  } else {
    const ax = form.A.apply(x);
    q += 0.5 * dotN(x, ax);
  }
  return q + dotN(form.b, x) + form.c;
}

export function gradientQuadraticForm(form: Readonly<QuadraticForm>, x: ReadonlyArray<number>): number[] {
  const ax = form.A.apply(x);
  const out = new Array<number>(form.dim);
  for (let i = 0; i < form.dim; i++) out[i] = ax[i] + form.b[i];
  return out;
}

export function sumQuadraticForms(forms: ReadonlyArray<QuadraticForm>, dim: number): QuadraticForm {
  if (forms.length === 0) return { dim, A: zeroOperator(dim), b: new Array<number>(dim).fill(0), c: 0 };
  const entries: SymmetricEntry[] = [];
  const b = new Array<number>(dim).fill(0);
  let c = 0;
  for (let i = 0; i < forms.length; i++) {
    const f = forms[i];
    const e = f.A.entries;
    if (e) {
      for (let k = 0; k < e.length; k++) entries.push(e[k]);
    } else {
      for (let d = 0; d < dim; d++) {
        const basis = new Array<number>(dim).fill(0);
        basis[d] = 1;
        const col = f.A.apply(basis);
        for (let j = d; j < dim; j++) {
          const v = col[j];
          if (Math.abs(v) > 1e-14) entries.push({ i: d, j, value: v });
        }
      }
    }
    for (let d = 0; d < dim; d++) b[d] += f.b[d];
    c += f.c;
  }
  return { dim, A: sparseSymmetricOperator(dim, entries), b, c };
}

export function makeLocalQuadraticFromValueGradDiag(
  xRef: ReadonlyArray<number>,
  valueAtRef: number,
  gradAtRef: ReadonlyArray<number>,
  diag: ReadonlyArray<number>
): QuadraticForm {
  const dim = xRef.length;
  const A = diagonalOperator(diag);
  const ax = A.apply(xRef);
  const b = new Array<number>(dim);
  for (let i = 0; i < dim; i++) b[i] = gradAtRef[i] - ax[i];
  const c = valueAtRef - 0.5 * dotN(xRef, ax) - dotN(b, xRef);
  return { dim, A, b, c };
}

export function linearConstraintAsQuadratic(
  id: string,
  sense: "eq" | "le",
  xRef: ReadonlyArray<number>,
  grad: ReadonlyArray<number>,
  valueAtRef: number,
  source: "exact" | "local"
): QuadraticConstraint {
  const dim = xRef.length;
  const b = [...grad];
  const c = valueAtRef - dotN(grad, xRef);
  return {
    id,
    sense,
    source,
    form: { dim, A: zeroOperator(dim), b, c },
  };
}
