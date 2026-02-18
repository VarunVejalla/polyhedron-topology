import { OptimizerSession } from "../session";
import type { OptimizationModel } from "../types";
import {
  consensusQcqpKernel,
  type ConsensusQcqpMemory,
  type ConsensusQcqpParams,
  type ConsensusQcqpState,
} from "../kernels/consensusQcqpKernel";
import { buildQuadraticSurrogateModel, type SurrogateQcqpModel } from "../surrogateQcqp";

export type SequentialConsensusParams = {
  rho: number;
  proximalWeight: number;
  linearSolveShift: number;
  qcqpTol: number;
  qcqpMaxNewtonIters: number;
  relinearizeEvery: number;
  innerIterationsPerOuter: number;
};

export class SequentialQuadraticConsensusSolver {
  private baseModel: OptimizationModel;
  private params: SequentialConsensusParams;
  private session: OptimizerSession<ConsensusQcqpState, ConsensusQcqpParams, ConsensusQcqpMemory, SurrogateQcqpModel>;
  private stepsSinceRelinearize = 0;

  constructor(baseModel: OptimizationModel, x0: number[], params: SequentialConsensusParams) {
    this.baseModel = baseModel;
    this.params = { ...params };
    this.session = new OptimizerSession({
      kernel: consensusQcqpKernel,
      model: buildQuadraticSurrogateModel(this.baseModel, x0),
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
        this.session.setModel(buildQuadraticSurrogateModel(this.baseModel, this.session.getStateRef().x));
        this.stepsSinceRelinearize = 0;
      }
      this.session.step(innerPerOuter);
      this.stepsSinceRelinearize += 1;
    }
  }
}
