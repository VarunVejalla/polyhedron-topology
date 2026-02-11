import type { Vec3 } from "../math/types";
import type { HandleSet, IProjector } from "./index";
import { bestFitPlanePCA, type Plane } from "../geom/plane";
export type AlternatingParams = {
  wFree: number;
  wHandle: number;
};

function solve3x3CholeskySPD(M: number[][], rhs: Vec3, fallback: Vec3): Vec3 {
  // Cholesky factorization for 3x3 SPD matrix M = L L^T.
  // Returns fallback if M is not numerically SPD.
  const a00 = M[0][0], a01 = M[0][1], a02 = M[0][2];
  const a11 = M[1][1], a12 = M[1][2];
  const a22 = M[2][2];

  if (!Number.isFinite(a00) || !Number.isFinite(a01) || !Number.isFinite(a02) ||
      !Number.isFinite(a11) || !Number.isFinite(a12) || !Number.isFinite(a22)) {
    return fallback;
  }

  // L00
  if (a00 <= 0) return fallback;
  const L00 = Math.sqrt(a00);

  // L10, L11
  const L10 = a01 / L00;
  const s11 = a11 - L10 * L10;
  if (s11 <= 0) return fallback;
  const L11 = Math.sqrt(s11);

  // L20, L21, L22
  const L20 = a02 / L00;
  const L21 = (a12 - L20 * L10) / L11;
  const s22 = a22 - L20 * L20 - L21 * L21;
  if (s22 <= 0) return fallback;
  const L22 = Math.sqrt(s22);

  // Solve L y = rhs
  const y0 = rhs[0] / L00;
  const y1 = (rhs[1] - L10 * y0) / L11;
  const y2 = (rhs[2] - L20 * y0 - L21 * y1) / L22;

  // Solve L^T x = y
  const x2 = y2 / L22;
  const x1 = (y1 - L21 * x2) / L11;
  const x0 = (y0 - L10 * x1 - L20 * x2) / L00;

  const out: Vec3 = [x0, x1, x2];
  if (!Number.isFinite(out[0]) || !Number.isFinite(out[1]) || !Number.isFinite(out[2])) return fallback;
  return out;
}


export class AlternatingPlanarProjector implements IProjector {
  private faces: number[][];
  private x0: Vec3[] = [];
  private x: Vec3[] = [];
  private params: AlternatingParams;
  private handles: HandleSet = { targets: new Map() };

  private inc: number[][] = [];
  private planes: Plane[] = [];
  private prevFaceNormals: Array<Vec3 | undefined> = [];
  private lastTotalViolation = 0;
  private xNew: Vec3[] = [];

  constructor(faces: number[][], x0: Vec3[], params: AlternatingParams) {
    this.faces = faces.map((f) => [...f]);
    this.params = { ...params };
    this.reset(x0);
  }

  reset(x0: Vec3[]) {
    this.x0 = x0.map((p) => [...p] as Vec3);
    this.x = x0.map((p) => [...p] as Vec3);
    this.xNew = x0.map((p) => [...p] as Vec3);

    this.inc = Array.from({ length: this.x.length }, () => []);
    for (let fi = 0; fi < this.faces.length; fi++) {
      for (const vi of this.faces[fi]) this.inc[vi].push(fi);
    }

    this.prevFaceNormals = new Array(this.faces.length).fill(undefined);
    this.planes = this.faces.map((f, fi) => {
      const pts = f.map((vi) => this.x[vi]);
      const plane = bestFitPlanePCA(pts, this.prevFaceNormals[fi]);
      this.prevFaceNormals[fi] = plane.n;
      return plane;
    });

    this.lastTotalViolation = this.computeTotalViolation();
  }

  setHandles(handles: HandleSet) {
    this.handles = handles;
  }

  setParams(next: Partial<AlternatingParams>) {
    this.params = { ...this.params, ...next };
  }


  private computeTotalViolation(): number {
    let total = 0;
    for (let fi = 0; fi < this.faces.length; fi++) {
      const { n, b } = this.planes[fi];
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
    const { wFree, wHandle } = this.params;

    for (let it = 0; it < iterations; it++) {
      // 1) Fit planes from current vertex positions (with sign continuity)
      for (let fi = 0; fi < this.faces.length; fi++) {
        const pts = this.faces[fi].map((vi) => this.x[vi]);
        const plane = bestFitPlanePCA(pts, this.prevFaceNormals[fi]);
        this.prevFaceNormals[fi] = plane.n;
        this.planes[fi] = plane;
      }

      // 2) For each vertex, solve a tiny LS system that tries to satisfy all incident planes.
      const xNew = this.xNew;
      for (let i = 0; i < this.x.length; i++) {
        const isHandle = this.handles.targets.has(i);
        const w = isHandle ? wHandle : wFree;
        const d = isHandle ? (this.handles.targets.get(i) as Vec3) : this.x0[i];

        // Minimize sum_f (n_f·x - b_f)^2 + w ||x - d||^2
        let A00 = w, A01 = 0, A02 = 0, A11 = w, A12 = 0, A22 = w;
        let r0 = w * d[0], r1 = w * d[1], r2 = w * d[2];

        for (const fi of this.inc[i]) {
          const { n, b } = this.planes[fi];
          A00 += n[0] * n[0]; A01 += n[0] * n[1]; A02 += n[0] * n[2];
          A11 += n[1] * n[1]; A12 += n[1] * n[2];
          A22 += n[2] * n[2];
          r0 += b * n[0]; r1 += b * n[1]; r2 += b * n[2];
        }

        const M = [
          [A00, A01, A02],
          [A01, A11, A12],
          [A02, A12, A22],
        ];
        const rhs: Vec3 = [r0, r1, r2];

        // Defensive fallback: previous position (NOT the origin).
        const sol = solve3x3CholeskySPD(M, rhs, this.x[i]);
        xNew[i][0] = sol[0];
        xNew[i][1] = sol[1];
        xNew[i][2] = sol[2];
      }

      const tmp = this.x;
      this.x = xNew;
      this.xNew = tmp;
      this.lastTotalViolation = this.computeTotalViolation();
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
