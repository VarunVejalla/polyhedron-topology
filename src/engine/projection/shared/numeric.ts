export function dotN(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function normN(a: ReadonlyArray<number>): number {
  return Math.sqrt(Math.max(0, dotN(a, a)));
}

export function solveCG(
  applyA: (p: number[]) => number[],
  b: ReadonlyArray<number>,
  maxIters: number,
  tol: number
): number[] {
  const n = b.length;
  const x = new Array<number>(n).fill(0);
  const r = Array.from(b);
  const p = r.slice();
  let rr = dotN(r, r);
  if (Math.sqrt(rr) <= tol) return x;

  for (let it = 0; it < maxIters; it++) {
    const Ap = applyA(p);
    const pAp = Math.max(1e-20, dotN(p, Ap));
    const alpha = rr / pAp;
    for (let i = 0; i < n; i++) x[i] += alpha * p[i];
    for (let i = 0; i < n; i++) r[i] -= alpha * Ap[i];
    const rrNew = dotN(r, r);
    if (Math.sqrt(rrNew) <= tol) break;
    const beta = rrNew / Math.max(1e-20, rr);
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    rr = rrNew;
  }
  return x;
}
