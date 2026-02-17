import type {
  OptimizerHyperParams,
  OptimizerKernel,
  OptimizerState,
  PieceSelector,
  ProblemBuilder,
  StepReport,
} from "./types";

type ControllerState<Ctx> = {
  context: Ctx;
  pieceKey: string;
};

const STATIC_PIECE_KEY = "__static_piece__";

export class PiecewiseOptimizationController<Ctx = unknown> {
  private readonly builder: ProblemBuilder<Ctx>;
  private readonly kernel: OptimizerKernel;
  private readonly selector: PieceSelector<Ctx> | null;
  private hp: OptimizerHyperParams;
  private state: OptimizerState;
  private problem;
  private ctl: ControllerState<Ctx>;

  constructor(args: {
    builder: ProblemBuilder<Ctx>;
    kernel: OptimizerKernel;
    x0: ReadonlyArray<number>;
    hyperParams: OptimizerHyperParams;
    selector?: PieceSelector<Ctx> | null;
  }) {
    this.builder = args.builder;
    this.kernel = args.kernel;
    this.selector = args.selector ?? null;
    this.hp = { ...args.hyperParams };
    const context = this.builder.initializeContext
      ? this.builder.initializeContext(args.x0, this.hp)
      : (undefined as Ctx);
    const pieceKey = this.computePieceKey(args.x0, context);
    const problem = this.builder.buildProblem(args.x0, context, this.hp);
    this.state = this.kernel.initialize(problem, args.x0, this.hp);
    this.problem = problem;
    this.ctl = { context, pieceKey };
  }

  getXRef(): ReadonlyArray<number> {
    return this.state.x;
  }

  getMutableState(): OptimizerState {
    return this.state;
  }

  getHyperParams(): Readonly<OptimizerHyperParams> {
    return this.hp;
  }

  setHyperParams(next: Partial<OptimizerHyperParams>) {
    this.hp = { ...this.hp, ...next };
  }

  reset(x0: ReadonlyArray<number>) {
    const context = this.builder.initializeContext
      ? this.builder.initializeContext(x0, this.hp)
      : this.ctl.context;
    const pieceKey = this.computePieceKey(x0, context);
    this.problem = this.builder.buildProblem(x0, context, this.hp);
    this.state = this.kernel.initialize(this.problem, x0, this.hp);
    this.ctl = { context, pieceKey };
  }

  step(iterations: number): StepReport {
    const out: StepReport = {
      attempted: 0,
      accepted: 0,
      lastAlpha: 0,
      eqResidualL2: 0,
      leViolationMax: 0,
      objective: 0,
    };
    if (iterations <= 0) return out;

    for (let i = 0; i < iterations; i++) {
      if (this.builder.updateContext) {
        this.ctl.context = this.builder.updateContext(this.state.x, this.ctl.context, this.hp);
      }

      const nextPiece = this.computePieceKey(this.state.x, this.ctl.context);
      if (nextPiece !== this.ctl.pieceKey) {
        this.problem = this.builder.buildProblem(this.state.x, this.ctl.context, this.hp);
        this.kernel.rebindProblem(this.problem, this.state, this.hp);
        this.ctl.pieceKey = nextPiece;
      }

      const report = this.kernel.step(this.problem, this.state, this.hp, 1);
      out.attempted += report.attempted;
      out.accepted += report.accepted;
      out.lastAlpha = report.lastAlpha;
      out.eqResidualL2 = report.eqResidualL2;
      out.leViolationMax = report.leViolationMax;
      out.objective = report.objective;

      if (report.accepted === 0) break;
    }

    return out;
  }

  private computePieceKey(x: ReadonlyArray<number>, context: Ctx): string {
    if (!this.selector) return STATIC_PIECE_KEY;
    return this.selector.keyAt(x, context, this.hp);
  }
}

