import type { OptimizationModel, OptimizerKernel } from "./types";

export class OptimizerSession<State, Params, Memory, Model = OptimizationModel> {
  private model: Model;
  private state: State;
  private params: Params;
  private memory: Memory;
  private kernel: OptimizerKernel<State, Params, Memory, Model>;

  constructor(args: {
    kernel: OptimizerKernel<State, Params, Memory, Model>;
    model: Model;
    initialState: State;
    params: Params;
    resume?: Memory;
  }) {
    this.kernel = args.kernel;
    this.model = args.model;
    this.state = args.initialState;
    this.params = args.params;
    this.memory = this.kernel.initialize({
      model: this.model,
      state: this.state,
      params: this.params,
      resume: args.resume,
    });
  }

  getStateRef(): Readonly<State> {
    return this.state;
  }

  getMutableState(): State {
    return this.state;
  }

  setState(next: State): void {
    this.state = next;
    this.memory = this.kernel.initialize({
      model: this.model,
      state: this.state,
      params: this.params,
      resume: this.memory,
    });
  }

  setModel(next: Model): void {
    this.model = next;
    this.memory = this.kernel.initialize({
      model: this.model,
      state: this.state,
      params: this.params,
      resume: this.memory,
    });
  }

  setParams(next: Partial<Params>): void {
    this.params = { ...this.params, ...next };
  }

  getParams(): Readonly<Params> {
    return this.params;
  }

  getMemoryRef(): Readonly<Memory> {
    return this.memory;
  }

  snapshotParams(): Params {
    return { ...this.params };
  }

  snapshotMemory(): Memory {
    return structuredClone(this.memory);
  }

  step(iterations: number): void {
    if (iterations <= 0) return;
    this.kernel.step({
      model: this.model,
      state: this.state,
      params: this.params,
      memory: this.memory,
      iterations,
    });
  }
}
