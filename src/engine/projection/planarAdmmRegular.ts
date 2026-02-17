import type { Vec3 } from "../math/types";
import { bestFitPlanePCA, type Plane } from "../geom/plane";
import type { HandleSet, IProjector } from "./index";

export type ADMMRegularParams = {
  rho: number;
  wFree: number;
  wHandle: number;
  lambdaReg: number;
  epsArea?: number;
  xInnerSteps?: number;
  fdStep?: number;
  regStepScale?: number;
};

export class ADMMRegularPlanarProjector implements IProjector {
  private faces: number[][];
  private x0: Vec3[] = [];
  private x: Vec3[] = [];
  private params: ADMMRegularParams;
  private handles: HandleSet = { targets: new Map() };

  // per-face y and u (same shape as face vertex list)
  private y: Vec3[][] = [];
  private u: Vec3[][] = [];

  // incidence: vertex -> list of (faceIndex, localIndex)
  private inc: Array<Array<{ fi: number; li: number }>> = [];
  private vertexFaces: number[][] = [];

  // normal continuity for planar prox
  private prevFaceNormals: Array<Vec3 | undefined> = [];

  // cached per-face plane fits from the most recent y-update
  private facePlanes: Plane[] = [];
  private lastTotalViolation = 0;

  // per-face temporary buffer v = x + u (same shape as face vertex list)
  private vbuf: Vec3[][] = [];

  constructor(faces: number[][], x0: Vec3[], params: ADMMRegularParams) {
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

    this.vertexFaces = Array.from({ length: this.x.length }, () => []);
    for (let vi = 0; vi < this.inc.length; vi++) {
      const fs = new Set<number>();
      for (const it of this.inc[vi]) fs.add(it.fi);
      this.vertexFaces[vi] = [...fs];
    }

    this.prevFaceNormals = new Array(this.faces.length).fill(undefined);
    this.vbuf = this.faces.map((f) => f.map(() => [0, 0, 0] as Vec3));

    this.facePlanes = this.faces.map((f, fi) => {
      const pts: Vec3[] = f.map((vi) => this.x[vi]);
      const plane = bestFitPlanePCA(pts, this.prevFaceNormals[fi]);
      this.prevFaceNormals[fi] = plane.n;
      return plane;
    });

    this.lastTotalViolation = this.computeTotalViolationFromCachedPlanes(this.x);
  }

  setHandles(handles: HandleSet) {
    this.handles = handles;
  }

  setParams(next: Partial<ADMMRegularParams>) {
    this.params = { ...this.params, ...next };
  }

  private computeTotalViolationFromCachedPlanes(positions: ReadonlyArray<Vec3>): number {
    let total = 0;
    for (let fi = 0; fi < this.faces.length; fi++) {
      const plane = this.facePlanes[fi];
      const n = plane.n;
      const b = plane.b;
      const face = this.faces[fi];
      for (let k = 0; k < face.length; k++) {
        const p = positions[face[k]];
        const d = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - b;
        total += Math.abs(d);
      }
    }
    return total;
  }

  private faceRegularity(fi: number, positions: ReadonlyArray<Vec3>, epsArea: number): number {
    const face = this.faces[fi];
    const nSides = face.length;
    if (nSides < 3) return 0;

    const pts: Vec3[] = face.map((vi) => positions[vi]);
    const plane = bestFitPlanePCA(pts, this.prevFaceNormals[fi]);

    const c = plane.c;
    const n = plane.n;

    // Build a stable in-plane basis (e1, e2).
    let e1: Vec3 = [0, 0, 0];
    let found = false;
    for (let i = 1; i < pts.length; i++) {
      const dx = pts[i][0] - pts[0][0];
      const dy = pts[i][1] - pts[0][1];
      const dz = pts[i][2] - pts[0][2];
      const dot = dx * n[0] + dy * n[1] + dz * n[2];
      const px = dx - dot * n[0];
      const py = dy - dot * n[1];
      const pz = dz - dot * n[2];
      const norm = Math.hypot(px, py, pz);
      if (norm > 1e-12) {
        e1 = [px / norm, py / norm, pz / norm];
        found = true;
        break;
      }
    }
    if (!found) {
      // Degenerate face: no meaningful regularity signal.
      return 0;
    }

    const e2: Vec3 = [
      n[1] * e1[2] - n[2] * e1[1],
      n[2] * e1[0] - n[0] * e1[2],
      n[0] * e1[1] - n[1] * e1[0],
    ];

    const uv: Array<{ u: number; v: number }> = [];
    for (let i = 0; i < pts.length; i++) {
      const qx = pts[i][0] - c[0];
      const qy = pts[i][1] - c[1];
      const qz = pts[i][2] - c[2];
      uv.push({
        u: qx * e1[0] + qy * e1[1] + qz * e1[2],
        v: qx * e2[0] + qy * e2[1] + qz * e2[2],
      });
    }

    let perimeter = 0;
    let twiceArea = 0;
    for (let i = 0; i < nSides; i++) {
      const a = uv[i];
      const b = uv[(i + 1) % nSides];
      const du = b.u - a.u;
      const dv = b.v - a.v;
      perimeter += Math.hypot(du, dv);
      twiceArea += a.u * b.v - a.v * b.u;
    }

    const area = Math.max(Math.abs(0.5 * twiceArea), epsArea);
    const cReg = 4 * nSides * Math.tan(Math.PI / nSides);
    return (perimeter * perimeter) / (cReg * area) - 1;
  }

  private regularityEnergy(positions: ReadonlyArray<Vec3>, epsArea: number): number {
    let e = 0;
    for (let fi = 0; fi < this.faces.length; fi++) {
      const r = this.faceRegularity(fi, positions, epsArea);
      e += r * r;
    }
    return e;
  }

  private regularityGradient(positions: Vec3[], epsArea: number, fdStep: number): Vec3[] {
    const grad: Vec3[] = positions.map(() => [0, 0, 0]);

    for (let vi = 0; vi < positions.length; vi++) {
      const localFaces = this.vertexFaces[vi];
      if (localFaces.length === 0) continue;

      const localEnergy = (): number => {
        let out = 0;
        for (const fi of localFaces) {
          const r = this.faceRegularity(fi, positions, epsArea);
          out += r * r;
        }
        return out;
      };

      for (let axis = 0; axis < 3; axis++) {
        const old = positions[vi][axis];
        positions[vi][axis] = old + fdStep;
        const ep = localEnergy();
        positions[vi][axis] = old - fdStep;
        const em = localEnergy();
        positions[vi][axis] = old;
        grad[vi][axis] = (ep - em) / (2 * fdStep);
      }
    }

    return grad;
  }

  private phi(
    positions: ReadonlyArray<Vec3>,
    cTargets: Array<Array<Vec3>>,
    rho: number,
    lambdaReg: number,
    epsArea: number
  ): number {
    let e = 0;
    for (let i = 0; i < positions.length; i++) {
      const isHandle = this.handles.targets.has(i);
      const w = isHandle ? this.params.wHandle : this.params.wFree;
      const d = isHandle ? (this.handles.targets.get(i) as Vec3) : this.x0[i];
      const dx = positions[i][0] - d[0];
      const dy = positions[i][1] - d[1];
      const dz = positions[i][2] - d[2];
      e += w * (dx * dx + dy * dy + dz * dz);
    }

    for (let fi = 0; fi < this.faces.length; fi++) {
      const f = this.faces[fi];
      for (let li = 0; li < f.length; li++) {
        const vi = f[li];
        const c = cTargets[fi][li];
        const dx = positions[vi][0] - c[0];
        const dy = positions[vi][1] - c[1];
        const dz = positions[vi][2] - c[2];
        e += 0.5 * rho * (dx * dx + dy * dy + dz * dz);
      }
    }

    if (lambdaReg > 0) {
      e += lambdaReg * this.regularityEnergy(positions, epsArea);
    }

    return e;
  }

  step(iterations: number) {
    const rho = this.params.rho;
    const epsArea = Math.max(1e-12, this.params.epsArea ?? 1e-8);
    const xInnerSteps = Math.max(1, Math.floor(this.params.xInnerSteps ?? 1));
    const fdStep = Math.max(1e-8, this.params.fdStep ?? 1e-5);
    const regStepScale = Math.max(0, this.params.regStepScale ?? 1);
    const lambdaReg = Math.max(0, this.params.lambdaReg);

    const cTargets: Array<Array<Vec3>> = this.faces.map((f) => f.map(() => [0, 0, 0] as Vec3));

    for (let it = 0; it < iterations; it++) {
      // 1) Local planar projection block (y-update first).
      for (let fi = 0; fi < this.faces.length; fi++) {
        const f = this.faces[fi];
        const v = this.vbuf[fi];

        for (let li = 0; li < f.length; li++) {
          const vi = f[li];
          const x = this.x[vi];
          const u = this.u[fi][li];
          v[li][0] = x[0] + u[0];
          v[li][1] = x[1] + u[1];
          v[li][2] = x[2] + u[2];
        }

        const plane = bestFitPlanePCA(v, this.prevFaceNormals[fi]);
        this.prevFaceNormals[fi] = plane.n;
        this.facePlanes[fi] = plane;

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

      // Consensus targets c = y - u
      for (let fi = 0; fi < this.faces.length; fi++) {
        const f = this.faces[fi];
        for (let li = 0; li < f.length; li++) {
          const y = this.y[fi][li];
          const u = this.u[fi][li];
          cTargets[fi][li][0] = y[0] - u[0];
          cTargets[fi][li][1] = y[1] - u[1];
          cTargets[fi][li][2] = y[2] - u[2];
        }
      }

      // 2) Global nonlinear x-step (prox-linear inner iterations).
      for (let inner = 0; inner < xInnerSteps; inner++) {
        const gradReg = lambdaReg > 0 ? this.regularityGradient(this.x, epsArea, fdStep) : this.x.map(() => [0, 0, 0] as Vec3);

        const xOld = this.x.map((p) => [p[0], p[1], p[2]] as Vec3);
        const xCandidate = this.x.map((p) => [p[0], p[1], p[2]] as Vec3);

        for (let i = 0; i < this.x.length; i++) {
          const deg = this.inc[i].length;
          const isHandle = this.handles.targets.has(i);
          const w = isHandle ? this.params.wHandle : this.params.wFree;
          const d = isHandle ? (this.handles.targets.get(i) as Vec3) : this.x0[i];

          let sum0 = 0, sum1 = 0, sum2 = 0;
          for (let k = 0; k < deg; k++) {
            const { fi, li } = this.inc[i][k];
            const c = cTargets[fi][li];
            sum0 += c[0];
            sum1 += c[1];
            sum2 += c[2];
          }

          const denom = w + rho * deg;
          const inv = 1 / Math.max(1e-12, denom);
          xCandidate[i][0] = (w * d[0] + rho * sum0) * inv - regStepScale * lambdaReg * gradReg[i][0] * inv;
          xCandidate[i][1] = (w * d[1] + rho * sum1) * inv - regStepScale * lambdaReg * gradReg[i][1] * inv;
          xCandidate[i][2] = (w * d[2] + rho * sum2) * inv - regStepScale * lambdaReg * gradReg[i][2] * inv;
        }

        if (lambdaReg > 0) {
          // Backtracking blend to keep the nonlinear step stable.
          const phiOld = this.phi(xOld, cTargets, rho, lambdaReg, epsArea);
          let alpha = 1;
          let accepted = false;
          for (let bt = 0; bt < 6; bt++) {
            for (let i = 0; i < this.x.length; i++) {
              this.x[i][0] = xOld[i][0] + alpha * (xCandidate[i][0] - xOld[i][0]);
              this.x[i][1] = xOld[i][1] + alpha * (xCandidate[i][1] - xOld[i][1]);
              this.x[i][2] = xOld[i][2] + alpha * (xCandidate[i][2] - xOld[i][2]);
            }
            const phiNew = this.phi(this.x, cTargets, rho, lambdaReg, epsArea);
            if (phiNew <= phiOld) {
              accepted = true;
              break;
            }
            alpha *= 0.5;
          }
          if (!accepted) {
            for (let i = 0; i < this.x.length; i++) {
              this.x[i][0] = xCandidate[i][0];
              this.x[i][1] = xCandidate[i][1];
              this.x[i][2] = xCandidate[i][2];
            }
          }
        } else {
          for (let i = 0; i < this.x.length; i++) {
            this.x[i][0] = xCandidate[i][0];
            this.x[i][1] = xCandidate[i][1];
            this.x[i][2] = xCandidate[i][2];
          }
        }
      }

      // 3) Dual update: u += x - y
      for (let fi = 0; fi < this.faces.length; fi++) {
        const f = this.faces[fi];
        for (let li = 0; li < f.length; li++) {
          const vi = f[li];
          const u = this.u[fi][li];
          const x = this.x[vi];
          const y = this.y[fi][li];
          u[0] += x[0] - y[0];
          u[1] += x[1] - y[1];
          u[2] += x[2] - y[2];
        }
      }

      this.lastTotalViolation = this.computeTotalViolationFromCachedPlanes(this.x);
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
