import type { Vec3 } from "../math/types";
import { bestFitPlanePCA, type Plane } from "../geom/plane";
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

    this.y = this.faces.map((f) => f.map((vi) => [...this.x[vi]] as Vec3));
    this.u = this.faces.map((f) => f.map(() => [0, 0, 0] as Vec3));

    this.inc = Array.from({ length: this.x.length }, () => []);
    for (let fi = 0; fi < this.faces.length; fi++) {
      const f = this.faces[fi];
      for (let li = 0; li < f.length; li++) {
        const vi = f[li];
        this.inc[vi].push({ fi, li });
      }
    }

    this.prevFaceNormals = new Array(this.faces.length).fill(undefined);

    // Allocate reusable per-face buffers.
    this.vbuf = this.faces.map((f) => f.map(() => [0, 0, 0] as Vec3));

    // Initialize cached planes/diagnostics from the initial positions.
    this.facePlanes = this.faces.map((f, fi) => {
      const pts: Vec3[] = f.map((vi) => this.x[vi]);
      const plane = bestFitPlanePCA(pts, this.prevFaceNormals[fi]);
      this.prevFaceNormals[fi] = plane.n;
      return plane;
    });

    this.lastTotalViolation = this.computeTotalViolationFromCachedPlanes();
  }

  setHandles(handles: HandleSet) {
    this.handles = handles;
  }

  setParams(next: Partial<ADMMParams>) {
    this.params = { ...this.params, ...next };
  }

  private computeTotalViolationFromCachedPlanes(): number {
    let total = 0;
    for (let fi = 0; fi < this.faces.length; fi++) {
      const plane = this.facePlanes[fi];
      const n = plane.n;
      const b = plane.b;
      const face = this.faces[fi];
      for (let k = 0; k < face.length; k++) {
        const p = this.x[face[k]];
        const d = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - b;
        total += Math.abs(d);
      }
    }
    return total;
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

      // y-update: per face planar prox (with normal sign continuity)
      for (let fi = 0; fi < this.faces.length; fi++) {
        const f = this.faces[fi];
        const v = this.vbuf[fi];

        // v = x + u (write into reusable buffer)
        for (let li = 0; li < f.length; li++) {
          const vi = f[li];
          const x = this.x[vi];
          const u = this.u[fi][li];
          v[li][0] = x[0] + u[0];
          v[li][1] = x[1] + u[1];
          v[li][2] = x[2] + u[2];
        }

        // Fit plane to v (with sign continuity), cache it, then prox-project.
        const plane = bestFitPlanePCA(v, this.prevFaceNormals[fi]);
        this.prevFaceNormals[fi] = plane.n;
        this.facePlanes[fi] = plane;

        // Project v onto the fitted plane (in-place write into y; no extra allocations).
        const yfi = this.y[fi];
        const n = plane.n;
        const b = plane.b;
        for (let li = 0; li < f.length; li++) {
          const p = v[li];
          const t = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - b;
          yfi[li][0] = p[0] - n[0] * t;
          yfi[li][1] = p[1] - n[1] * t;
          yfi[li][2] = p[2] - n[2] * t;
        }
      }

      // u-update: u += x - y
      for (let fi = 0; fi < this.faces.length; fi++) {
        const f = this.faces[fi];
        for (let li = 0; li < f.length; li++) {
          const vi = f[li];
          const u = this.u[fi][li];
          const x = this.x[vi];
          const y = this.y[fi][li];
          u[0] += (x[0] - y[0]);
          u[1] += (x[1] - y[1]);
          u[2] += (x[2] - y[2]);
        }
      }

      // Cache diagnostics from the planes we just fit.
      this.lastTotalViolation = this.computeTotalViolationFromCachedPlanes();
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
