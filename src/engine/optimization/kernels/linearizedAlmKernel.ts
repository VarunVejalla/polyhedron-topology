import { dotN, normN, solveCG } from "../../projection/shared/numeric";
import {
  evaluateQuadraticForm,
  gradientQuadraticForm,
  sumQuadraticForms,
} from "../quadratic";
import type {
  OptimizationProblem,
  OptimizerHyperParams,
  OptimizerKernel,
  OptimizerState,
  QuadraticConstraint,
  QuadraticForm,
  StepReport,
} from "../types";

type LinearizedConstraints = {
  eq: Array<{ id: string; value: number; grad: number[] }>;
  le: Array<{ id: string; value: number; grad: number[] }>;
};

function defaults(hp: Readonly<OptimizerHyperParams>) {
  return {
    rho: hp.rho,
    tau: Math.max(1e-10, hp.tau ?? 1e-6),
    cgIters: Math.max(4, Math.floor(hp.cgIters ?? 80)),
    cgTol: Math.max(1e-10, hp.cgTol ?? 1e-6),
    lineSearchC1: hp.lineSearchC1 ?? 1e-4,
    lineSearchShrink: Math.min(0.95, Math.max(0.1, hp.lineSearchShrink ?? 0.5)),
    lineSearchMaxSteps: Math.max(1, Math.floor(hp.lineSearchMaxSteps ?? 8)),
    adaptRho: hp.adaptRho ?? false,
    rhoIncrease: Math.max(1.01, hp.rhoIncrease ?? 2),
    rhoDecrease: Math.max(1.01, hp.rhoDecrease ?? 2),
    rhoResidualRatio: Math.max(1.1, hp.rhoResidualRatio ?? 10),
    rhoMin: Math.max(1e-8, hp.rhoMin ?? 1e-3),
    rhoMax: Math.max(Math.max(1e-8, hp.rhoMin ?? 1e-3), hp.rhoMax ?? 1e8),
  };
}

function allEq(problem: Readonly<OptimizationProblem>): QuadraticConstraint[] {
  return [...problem.exactEq, ...problem.localEq];
}

function allLe(problem: Readonly<OptimizationProblem>): QuadraticConstraint[] {
  return [...problem.exactLe, ...problem.localLe];
}

function objectiveForm(problem: Readonly<OptimizationProblem>): QuadraticForm {
  const forms: QuadraticForm[] = [];
  if (problem.metric) forms.push(problem.metric);
  if (problem.regularizer) forms.push(problem.regularizer);
  return sumQuadraticForms(forms, problem.dim);
}

function linearizeConstraints(problem: Readonly<OptimizationProblem>, x: ReadonlyArray<number>): LinearizedConstraints {
  const eq = allEq(problem).map((c) => ({
    id: c.id,
    value: evaluateQuadraticForm(c.form, x),
    grad: gradientQuadraticForm(c.form, x),
  }));
  const le = allLe(problem).map((c) => ({
    id: c.id,
    value: evaluateQuadraticForm(c.form, x),
    grad: gradientQuadraticForm(c.form, x),
  }));
  return { eq, le };
}

function merit(
  problem: Readonly<OptimizationProblem>,
  x: ReadonlyArray<number>,
  state: Readonly<OptimizerState>,
  rho: number
): number {
  const objective = problem.objectiveValueOverride
    ? problem.objectiveValueOverride(x)
    : evaluateQuadraticForm(objectiveForm(problem), x);
  const lin = linearizeConstraints(problem, x);
  let pen = 0;
  for (let i = 0; i < lin.eq.length; i++) {
    const t = lin.eq[i].value + (state.eqDual[i] ?? 0);
    pen += t * t;
  }
  for (let i = 0; i < lin.le.length; i++) {
    const t = Math.max(0, lin.le[i].value + (state.leDual[i] ?? 0));
    pen += t * t;
  }
  return objective + 0.5 * rho * pen;
}

function applyJT(
  eq: ReadonlyArray<{ grad: ReadonlyArray<number> }>,
  le: ReadonlyArray<{ grad: ReadonlyArray<number> }>,
  eqWeights: ReadonlyArray<number>,
  leWeights: ReadonlyArray<number>,
  dim: number
): number[] {
  const out = new Array<number>(dim).fill(0);
  for (let i = 0; i < eq.length; i++) {
    const wi = eqWeights[i];
    if (wi === 0) continue;
    const g = eq[i].grad;
    for (let k = 0; k < dim; k++) out[k] += wi * g[k];
  }
  for (let i = 0; i < le.length; i++) {
    const wi = leWeights[i];
    if (wi === 0) continue;
    const g = le[i].grad;
    for (let k = 0; k < dim; k++) out[k] += wi * g[k];
  }
  return out;
}

function activeLeWeights(
  le: ReadonlyArray<{ value: number }>,
  dual: ReadonlyArray<number>
): number[] {
  const out = new Array<number>(le.length);
  for (let i = 0; i < le.length; i++) out[i] = Math.max(0, le[i].value + (dual[i] ?? 0));
  return out;
}

function ensureDualSizing(state: OptimizerState, problem: Readonly<OptimizationProblem>) {
  const eqIds = allEq(problem).map((c) => c.id);
  const leIds = allLe(problem).map((c) => c.id);
  const eqMap = new Map<string, number>();
  const leMap = new Map<string, number>();
  for (let i = 0; i < state.eqIds.length; i++) eqMap.set(state.eqIds[i], state.eqDual[i] ?? 0);
  for (let i = 0; i < state.leIds.length; i++) leMap.set(state.leIds[i], state.leDual[i] ?? 0);
  state.eqIds = eqIds;
  state.leIds = leIds;
  state.eqDual = eqIds.map((id) => eqMap.get(id) ?? 0);
  state.leDual = leIds.map((id) => leMap.get(id) ?? 0);
}

export class LinearizedAlmKernel implements OptimizerKernel {
  readonly id = "linearized_alm";

  initialize(
    problem: Readonly<OptimizationProblem>,
    x0: ReadonlyArray<number>,
    hp: Readonly<OptimizerHyperParams>
  ): OptimizerState {
    const cfg = defaults(hp);
    const eqIds = allEq(problem).map((c) => c.id);
    const leIds = allLe(problem).map((c) => c.id);
    return {
      x: [...x0],
      rho: cfg.rho,
      eqDual: new Array<number>(eqIds.length).fill(0),
      leDual: new Array<number>(leIds.length).fill(0),
      eqIds,
      leIds,
    };
  }

  rebindProblem(
    problem: Readonly<OptimizationProblem>,
    state: OptimizerState
  ) {
    ensureDualSizing(state, problem);
  }

  step(
    problem: Readonly<OptimizationProblem>,
    state: OptimizerState,
    hp: Readonly<OptimizerHyperParams>,
    iterations: number
  ): StepReport {
    const cfg = defaults(hp);
    if (state.rho <= 0) state.rho = cfg.rho;
    ensureDualSizing(state, problem);
    const HObj = objectiveForm(problem);
    const out: StepReport = {
      attempted: 0,
      accepted: 0,
      lastAlpha: 0,
      eqResidualL2: 0,
      leViolationMax: 0,
      objective: problem.objectiveValueOverride
        ? problem.objectiveValueOverride(state.x)
        : evaluateQuadraticForm(HObj, state.x),
    };
    if (iterations <= 0) return out;

    for (let it = 0; it < iterations; it++) {
      const lin = linearizeConstraints(problem, state.x);
      const objGrad = gradientQuadraticForm(HObj, state.x);

      const eqWeights = new Array<number>(lin.eq.length);
      for (let i = 0; i < lin.eq.length; i++) eqWeights[i] = lin.eq[i].value + state.eqDual[i];
      const leWeights = activeLeWeights(lin.le, state.leDual);

      const jtRes = applyJT(lin.eq, lin.le, eqWeights, leWeights, problem.dim);
      const rhs = new Array<number>(problem.dim);
      for (let i = 0; i < rhs.length; i++) rhs[i] = objGrad[i] + state.rho * jtRes[i];

      const applyA = (v: number[]): number[] => {
        const hv = HObj.A.apply(v);
        const eqJv = new Array<number>(lin.eq.length);
        const leJv = new Array<number>(lin.le.length);
        for (let i = 0; i < lin.eq.length; i++) eqJv[i] = dotN(lin.eq[i].grad, v);
        for (let i = 0; i < lin.le.length; i++) {
          const active = lin.le[i].value + state.leDual[i] > 0 ? 1 : 0;
          leJv[i] = active ? dotN(lin.le[i].grad, v) : 0;
        }
        const jtJv = applyJT(lin.eq, lin.le, eqJv, leJv, problem.dim);
        const outV = new Array<number>(problem.dim);
        for (let k = 0; k < outV.length; k++) outV[k] = hv[k] + cfg.tau * v[k] + state.rho * jtJv[k];
        return outV;
      };

      const delta = solveCG(applyA, rhs.map((v) => -v), cfg.cgIters, cfg.cgTol);
      const dirDeriv = dotN(rhs, delta);
      out.attempted += 1;
      if (!Number.isFinite(dirDeriv) || dirDeriv >= 0) break;

      const psi0 = merit(problem, state.x, state, state.rho);
      let alpha = 1;
      let accepted = false;
      const trial = state.x.slice();
      for (let ls = 0; ls < cfg.lineSearchMaxSteps; ls++) {
        for (let k = 0; k < trial.length; k++) trial[k] = state.x[k] + alpha * delta[k];
        const psiTrial = merit(problem, trial, state, state.rho);
        if (psiTrial <= psi0 + cfg.lineSearchC1 * alpha * dirDeriv) {
          accepted = true;
          break;
        }
        alpha *= cfg.lineSearchShrink;
      }
      if (!accepted) break;

      const prevLin = lin;
      state.x = trial.slice();
      out.lastAlpha = alpha;
      out.accepted += 1;

      const newLin = linearizeConstraints(problem, state.x);
      for (let i = 0; i < newLin.eq.length; i++) state.eqDual[i] += newLin.eq[i].value;
      for (let i = 0; i < newLin.le.length; i++) {
        state.leDual[i] = Math.max(0, state.leDual[i] + newLin.le[i].value);
      }

      if (cfg.adaptRho) {
        const primalSqEq = newLin.eq.reduce((s, v) => s + v.value * v.value, 0);
        const primalSqLe = newLin.le.reduce((s, v) => {
          const vv = Math.max(0, v.value);
          return s + vv * vv;
        }, 0);
        const primal = Math.sqrt(primalSqEq + primalSqLe);
        const eqDc = newLin.eq.map((v, i) => v.value - prevLin.eq[i].value);
        const leDc = newLin.le.map((v, i) => {
          const a = Math.max(0, v.value);
          const b = Math.max(0, prevLin.le[i].value);
          return a - b;
        });
        const dualVec = applyJT(newLin.eq, newLin.le, eqDc, leDc, problem.dim);
        const dual = state.rho * normN(dualVec);
        let rhoNew = state.rho;
        if (primal > cfg.rhoResidualRatio * dual) rhoNew = Math.min(cfg.rhoMax, state.rho * cfg.rhoIncrease);
        else if (dual > cfg.rhoResidualRatio * primal) rhoNew = Math.max(cfg.rhoMin, state.rho / cfg.rhoDecrease);
        if (rhoNew !== state.rho) {
          const scale = state.rho / rhoNew;
          for (let i = 0; i < state.eqDual.length; i++) state.eqDual[i] *= scale;
          for (let i = 0; i < state.leDual.length; i++) state.leDual[i] *= scale;
          state.rho = rhoNew;
        }
      }

      out.eqResidualL2 = Math.sqrt(newLin.eq.reduce((s, v) => s + v.value * v.value, 0));
      out.leViolationMax = newLin.le.reduce((m, v) => Math.max(m, Math.max(0, v.value)), 0);
      out.objective = problem.objectiveValueOverride
        ? problem.objectiveValueOverride(state.x)
        : evaluateQuadraticForm(HObj, state.x);
    }

    return out;
  }
}
