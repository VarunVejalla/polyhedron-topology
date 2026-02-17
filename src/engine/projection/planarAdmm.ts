import type { Vec3 } from "../math/types";
import type { Plane } from "../geom/plane";
import {
  buildVertexIncidence,
  computePlanarityViolationFromPlanes,
  createPlanarFaceBuffers,
  fitFacePlanesFromPositions,
  updatePlanarDualBlock,
  updatePlanarYBlock,
} from "./admmPlanarShared";
import type { HandleSet, IProjector } from "./index";

export type ADMMParams = {
  rho: number;
  wFree: number;
  wHandle: number;
};

export class ADMMPlanarProjector implements IProjector {
  private faces: number[][];
  private x0: Vec3[] = [];
  private x: Vec3[] = [];
  private params: ADMMParams;
  private handles: HandleSet = { targets: new Map() };

  // per-face y and u (same shape as face vertex list)
  private y: Vec3[][] = [];
  private u: Vec3[][] = [];

  // incidence: vertex -> list of (faceIndex, localIndex)
  private inc: Array<Array<{ fi: number; li: number }>> = [];

  // normal continuity for planar prox
  private prevFaceNormals: Array<Vec3 | undefined> = [];

  // cached per-face plane fits from the most recent y-update
  private facePlanes: Plane[] = [];
  // cached diagnostics (max |n·x-b| over all faces)
  private lastTotalViolation = 0;

  // per-face temporary buffer v = x + u (same shape as face vertex list)
  private vbuf: Vec3[][] = [];

  constructor(faces: number[][], x0: Vec3[], params: ADMMParams) {
    this.faces = faces.map((f) => [...f]);
    this.params = { ...params };
    this.reset(x0);
  }

  reset(x0: Vec3[]) {
    this.x0 = x0.map((p) => [...p] as Vec3);
    this.x = x0.map((p) => [...p] as Vec3);

    const planar = createPlanarFaceBuffers(this.faces, this.x);
    this.y = planar.y;
    this.u = planar.u;
    this.vbuf = planar.vbuf;
    this.inc = buildVertexIncidence(this.faces, this.x.length);

    this.prevFaceNormals = new Array(this.faces.length).fill(undefined);

    // Initialize cached planes/diagnostics from the initial positions.
    this.facePlanes = fitFacePlanesFromPositions(this.faces, this.x, this.prevFaceNormals);
    this.lastTotalViolation = computePlanarityViolationFromPlanes(this.faces, this.x, this.facePlanes);
  }

  setHandles(handles: HandleSet) {
    this.handles = handles;
  }

  setParams(next: Partial<ADMMParams>) {
    this.params = { ...this.params, ...next };
  }

  step(iterations: number) {
    const { rho, wFree, wHandle } = this.params;

    for (let it = 0; it < iterations; it++) {
      // x-update (closed form per vertex). In-place update; depends only on (y,u) and data term.
      for (let i = 0; i < this.x.length; i++) {
        const deg = this.inc[i].length;
        const isHandle = this.handles.targets.has(i);
        const w = isHandle ? wHandle : wFree;
        const d = isHandle ? (this.handles.targets.get(i) as Vec3) : this.x0[i];

        let sum0 = 0, sum1 = 0, sum2 = 0;
        for (let k = 0; k < this.inc[i].length; k++) {
          const { fi, li } = this.inc[i][k];
          const y = this.y[fi][li];
          const u = this.u[fi][li];
          sum0 += (y[0] - u[0]);
          sum1 += (y[1] - u[1]);
          sum2 += (y[2] - u[2]);
        }

        const denom = w + rho * deg;
        const inv = 1 / denom;
        this.x[i][0] = (w * d[0] + rho * sum0) * inv;
        this.x[i][1] = (w * d[1] + rho * sum1) * inv;
        this.x[i][2] = (w * d[2] + rho * sum2) * inv;
      }

      updatePlanarYBlock(this.faces, this.x, this.u, this.vbuf, this.y, this.prevFaceNormals, this.facePlanes);
      updatePlanarDualBlock(this.faces, this.x, this.y, this.u);

      // Cache diagnostics from the planes we just fit.
      this.lastTotalViolation = computePlanarityViolationFromPlanes(this.faces, this.x, this.facePlanes);
    }
  }

  getPositionsRef(): ReadonlyArray<Vec3> {
    return this.x;
  }

  snapshotPositions(): Vec3[] {
    return this.x.map((p) => [...p] as Vec3);
  }

  diagnostics() {
    return { totalPlanarityViolation: this.lastTotalViolation };
  }
}
