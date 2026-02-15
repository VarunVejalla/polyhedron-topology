import type {
  DualUpdater,
  Globalizer,
  MetaModelBuilder,
  MetaState,
  MetaStats,
  PenaltyPolicy,
  PrimalEngine,
  StopPolicy,
} from "./types";

export function runMetaSolver(
  state: MetaState,
  builder: MetaModelBuilder,
  engine: PrimalEngine,
  globalizer: Globalizer,
  dualUpdater: DualUpdater,
  penaltyPolicy: PenaltyPolicy,
  stopPolicy: StopPolicy | undefined,
  iterations: number
): MetaStats {
  const stats: MetaStats = { attempted: 0, accepted: 0, lastAlpha: 0 };
  if (iterations <= 0) return stats;

  for (let i = 0; i < iterations; i++) {
    const model = builder.build(state);
    const proposal = engine.propose(state, model);
    const accepted = globalizer.accept(state, model, proposal);

    stats.attempted += 1;
    if (!accepted.accepted) break;
    stats.lastAlpha = accepted.alpha;

    state.y = accepted.yNext;
    stats.accepted += 1;

    const cNew = model.hard.evaluate(state.y);
    dualUpdater.update(state, cNew);
    penaltyPolicy.update(state, model, cNew);

    if (stopPolicy?.shouldStop(state, model, cNew, stats)) break;
  }

  return stats;
}
