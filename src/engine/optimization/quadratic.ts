import type { DenseMatrix, QuadraticForm, ScalarFunction } from "./types";

function dot(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  let out = 0;
  for (let i = 0; i < a.length; i++) out += a[i] * b[i];
  return out;
}

function matVec(m: DenseMatrix, x: ReadonlyArray<number>): number[] {
  const out = new Array<number>(x.length).fill(0);
  for (let i = 0; i < m.length; i++) {
    let row = 0;
    for (let j = 0; j < x.length; j++) row += (m[i]?.[j] ?? 0) * x[j];
    out[i] = row;
  }
  return out;
}

export function zeroQuadratic(dim: number): QuadraticForm {
  return {
    A: Array.from({ length: dim }, () => new Array<number>(dim).fill(0)),
    b: new Array<number>(dim).fill(0),
    c: 0,
  };
}

export function evaluateQuadratic(form: Readonly<QuadraticForm>, x: ReadonlyArray<number>): number {
  const ax = matVec(form.A, x);
  return 0.5 * dot(x, ax) + dot(form.b, x) + form.c;
}

export function quadraticizeAt(fn: ScalarFunction, x: ReadonlyArray<number>): QuadraticForm {
  const A = fn.hessian(x);
  const grad = fn.jacobian(x);
  const value = fn.evaluate(x);
  const Ax = matVec(A, x);
  const b = new Array<number>(x.length);
  for (let i = 0; i < x.length; i++) b[i] = grad[i] - Ax[i];
  const c = value - dot(grad, x) + 0.5 * dot(x, Ax);
  return { A, b, c };
}
