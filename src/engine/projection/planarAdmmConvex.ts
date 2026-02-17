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

export type ADMMConvexParams = {
  rho: number;
  wFree: number;
  wHandle: number;
  rhoConvex?: number;
  convexPasses?: number;
  convexEps?: number;
};

export class ADMMConvexPlanarProjector implements IProjector {
  private faces: number[][];
  private x0: Vec3[] = [];
  private x: Vec3[] = [];
  private params: ADMMConvexParams;
  private handles: HandleSet = { targets: new Map() };

  // Planarity block
  private y: Vec3[][] = [];
  private u: Vec3[][] = [];
  private inc: Array<Array<{ fi: number; li: number }>> = [];
  private prevFaceNormals: Array<Vec3 | undefined> = [];
  private facePlanes: Plane[] = [];
  private vbuf: Vec3[][] = [];

  // Convexity block
  private z: Vec3[] = [];
  private q: Vec3[] = [];
  private faceVertexSets: Array<Set<number>> = [];

  private lastTotalViolation = 0;

  constructor(faces: number[][], x0: Vec3[], params: ADMMConvexParams) {
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
    this.facePlanes = fitFacePlanesFromPositions(this.faces, this.x, this.prevFaceNormals);

    this.faceVertexSets = this.faces.map((f) => new Set<number>(f));

    this.z = this.x.map((p) => [...p] as Vec3);
    this.q = this.x.map(() => [0, 0, 0] as Vec3);

    this.lastTotalViolation = computePlanarityViolationFromPlanes(this.faces, this.x, this.facePlanes);
  }

  setHandles(handles: HandleSet) {
    this.handles = handles;
  }

  setParams(next: Partial<ADMMConvexParams>) {
    this.params = { ...this.params, ...next };
  }

  private orientPlanesOutward(): Array<{ n: Vec3; b: number }> {
    const c: Vec3 = [0, 0, 0];
    for (let i = 0; i < this.x.length; i++) {
      c[0] += this.x[i][0];
      c[1] += this.x[i][1];
      c[2] += this.x[i][2];
    }
    const invN = this.x.length > 0 ? 1 / this.x.length : 0;
    c[0] *= invN;
    c[1] *= invN;
    c[2] *= invN;

    const out: Array<{ n: Vec3; b: number }> = [];
    for (let fi = 0; fi < this.facePlanes.length; fi++) {
      const pl = this.facePlanes[fi];
      let n: Vec3 = [pl.n[0], pl.n[1], pl.n[2]];
      let b = pl.b;
      // For outward normals, an interior point should satisfy n·x <= b.
      const side = n[0] * c[0] + n[1] * c[1] + n[2] * c[2] - b;
      if (side > 0) {
        n = [-n[0], -n[1], -n[2]];
        b = -b;
      }
      out.push({ n, b });
    }
    return out;
  }

  private projectConvexHalfspaces(seed: ReadonlyArray<Vec3>, orientedPlanes: Array<{ n: Vec3; b: number }>) {
    const passes = Math.max(1, Math.floor(this.params.convexPasses ?? 2));
    const eps = Math.max(0, this.params.convexEps ?? 1e-6);

    // Initialize z from seed.
    for (let i = 0; i < this.z.length; i++) {
      this.z[i][0] = seed[i][0];
      this.z[i][1] = seed[i][1];
      this.z[i][2] = seed[i][2];
    }

    // Sequential half-space projection (POCS).
    for (let pass = 0; pass < passes; pass++) {
      for (let fi = 0; fi < this.faces.length; fi++) {
        const { n, b } = orientedPlanes[fi];
        const faceSet = this.faceVertexSets[fi];
        const rhs = b - eps;
        for (let vi = 0; vi < this.z.length; vi++) {
          if (faceSet.has(vi)) continue;
          const p = this.z[vi];
          const d = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - rhs;
          if (d <= 0) continue;
          p[0] -= d * n[0];
          p[1] -= d * n[1];
          p[2] -= d * n[2];
        }
      }
    }
  }

  step(iterations: number) {
    const rhoPlanar = this.params.rho;
    const rhoConvex = this.params.rhoConvex ?? this.params.rho;
    const { wFree, wHandle } = this.params;

    for (let it = 0; it < iterations; it++) {
      // 1) x-update: data + planar consensus + convex consensus.
      for (let i = 0; i < this.x.length; i++) {
        const deg = this.inc[i].length;
        const isHandle = this.handles.targets.has(i);
        const w = isHandle ? wHandle : wFree;
        const d = isHandle ? (this.handles.targets.get(i) as Vec3) : this.x0[i];

        let sum0 = 0, sum1 = 0, sum2 = 0;
        for (let k = 0; k < deg; k++) {
          const { fi, li } = this.inc[i][k];
          const y = this.y[fi][li];
          const u = this.u[fi][li];
          sum0 += y[0] - u[0];
          sum1 += y[1] - u[1];
          sum2 += y[2] - u[2];
        }

        const z = this.z[i];
        const q = this.q[i];
        sum0 += (rhoConvex / rhoPlanar) * (z[0] - q[0]);
        sum1 += (rhoConvex / rhoPlanar) * (z[1] - q[1]);
        sum2 += (rhoConvex / rhoPlanar) * (z[2] - q[2]);

        const denom = w + rhoPlanar * deg + rhoConvex;
        const inv = 1 / denom;
        this.x[i][0] = (w * d[0] + rhoPlanar * sum0) * inv;
        this.x[i][1] = (w * d[1] + rhoPlanar * sum1) * inv;
        this.x[i][2] = (w * d[2] + rhoPlanar * sum2) * inv;
      }

      updatePlanarYBlock(this.faces, this.x, this.u, this.vbuf, this.y, this.prevFaceNormals, this.facePlanes);

      // 3) z-update: project (x + q) onto convex half-space constraints.
      const seed: Vec3[] = this.x.map((p, i) => [p[0] + this.q[i][0], p[1] + this.q[i][1], p[2] + this.q[i][2]]);
      const orientedPlanes = this.orientPlanesOutward();
      this.projectConvexHalfspaces(seed, orientedPlanes);

      // 4) dual updates.
      // u += x - y
      updatePlanarDualBlock(this.faces, this.x, this.y, this.u);
      // q += x - z
      for (let i = 0; i < this.x.length; i++) {
        this.q[i][0] += this.x[i][0] - this.z[i][0];
        this.q[i][1] += this.x[i][1] - this.z[i][1];
        this.q[i][2] += this.x[i][2] - this.z[i][2];
      }

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

