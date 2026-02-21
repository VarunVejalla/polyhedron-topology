import type { Plane } from "../math/plane";
import type { Vec3 } from "../math/types";
import { v3 } from "../math/vec3";
import {
  BBOX_MIN_SCALE,
  CONVEX_HALFSPACE_EPS,
  EPS,
  INVERSE_DENOM_EPS_LOOSE,
  LEGACY_STEP_CAP_RATIO,
  MIN_RHO_LEGACY,
} from "../math/constants";
import {
  buildVertexIncidence,
  computePlanarityViolationFromPlanes,
  createPlanarFaceBuffers,
  fitFacePlanesFromPositions,
  updatePlanarDualBlock,
  updatePlanarYBlock,
} from "./admmPlanarShared";
import type { HandleSet, IProjector, ProjectionFlavor, ProjectorParams } from "./index";

type GuidedMemory = {
  incidence: Array<Array<{ fi: number; li: number }>>;
  y: Vec3[][];
  u: Vec3[][];
  vbuf: Vec3[][];
  z: Vec3[];
  q: Vec3[];
  faceVertexSets: Array<Set<number>>;
  prevFaceNormals: Array<Vec3 | undefined>;
  facePlanes: Plane[];
  totalPlanarityViolation: number;
};

function cloneVec3(p: ReadonlyArray<number>): Vec3 {
  return [p[0], p[1], p[2]];
}

function cloneVec3List(points: ReadonlyArray<ReadonlyArray<number>>): Vec3[] {
  return points.map((p) => cloneVec3(p));
}

function sanitizeFaces(faces: ReadonlyArray<ReadonlyArray<number>>, vertexCount: number): number[][] {
  const cleaned: number[][] = [];
  for (let fi = 0; fi < faces.length; fi++) {
    const face = faces[fi].filter((vi) => Number.isInteger(vi) && vi >= 0 && vi < vertexCount);
    if (face.length >= 3) cleaned.push([...face]);
  }
  return cleaned;
}

function bboxScale(points: ReadonlyArray<Vec3>): number {
  if (points.length === 0) return 1;
  let minx = points[0][0];
  let miny = points[0][1];
  let minz = points[0][2];
  let maxx = points[0][0];
  let maxy = points[0][1];
  let maxz = points[0][2];
  for (let i = 1; i < points.length; i++) {
    const p = points[i];
    if (p[0] < minx) minx = p[0];
    if (p[1] < miny) miny = p[1];
    if (p[2] < minz) minz = p[2];
    if (p[0] > maxx) maxx = p[0];
    if (p[1] > maxy) maxy = p[1];
    if (p[2] > maxz) maxz = p[2];
  }
  const dx = maxx - minx;
  const dy = maxy - miny;
  const dz = maxz - minz;
  return Math.max(BBOX_MIN_SCALE, Math.sqrt(dx * dx + dy * dy + dz * dz));
}

function orientPlanesOutward(positions: ReadonlyArray<Vec3>, planes: ReadonlyArray<Plane>): Array<{ n: Vec3; b: number }> {
  let center: Vec3 = [0, 0, 0];
  for (let i = 0; i < positions.length; i++) center = v3.add(center, positions[i]);
  center = positions.length > 0 ? v3.mul(center, 1 / positions.length) : center;

  return planes.map((plane) => {
    let n: Vec3 = [plane.n[0], plane.n[1], plane.n[2]];
    let b = plane.b;
    if (v3.dot(n, center) - b > 0) {
      n = v3.mul(n, -1);
      b = -b;
    }
    return { n, b };
  });
}

function projectConvexHalfspaces(
  faces: ReadonlyArray<ReadonlyArray<number>>,
  memory: GuidedMemory,
  seed: ReadonlyArray<Vec3>,
  passes: number,
  eps: number
): void {
  for (let i = 0; i < memory.z.length; i++) memory.z[i] = cloneVec3(seed[i]);
  const oriented = orientPlanesOutward(seed, memory.facePlanes);
  for (let pass = 0; pass < passes; pass++) {
    for (let fi = 0; fi < faces.length; fi++) {
      const faceSet = memory.faceVertexSets[fi];
      const { n, b } = oriented[fi];
      const rhs = b - eps;
      for (let vi = 0; vi < memory.z.length; vi++) {
        if (faceSet.has(vi)) continue;
        const d = v3.dot(n, memory.z[vi]) - rhs;
        if (d <= 0) continue;
        memory.z[vi] = v3.sub(memory.z[vi], v3.mul(n, d));
      }
    }
  }
}

function createMemory(faces: number[][], positions: Vec3[]): GuidedMemory {
  const planar = createPlanarFaceBuffers(faces, positions);
  const prevFaceNormals = new Array<Vec3 | undefined>(faces.length).fill(undefined);
  const facePlanes = fitFacePlanesFromPositions(faces, positions, prevFaceNormals);
  return {
    incidence: buildVertexIncidence(faces, positions.length),
    y: planar.y,
    u: planar.u,
    vbuf: planar.vbuf,
    z: cloneVec3List(positions),
    q: positions.map(() => [0, 0, 0] as Vec3),
    faceVertexSets: faces.map((face) => new Set<number>(face)),
    prevFaceNormals,
    facePlanes,
    totalPlanarityViolation: computePlanarityViolationFromPlanes(faces, positions, facePlanes),
  };
}

export class GuidedAlmLegacyProjector implements IProjector {
  private readonly faces: number[][];
  private readonly flavor: ProjectionFlavor;
  private baseline: Vec3[];
  private positions: Vec3[];
  private handles = new Map<number, Vec3>();
  private params: ProjectorParams;
  private memory: GuidedMemory;

  constructor(facesArg: number[][], x0: Vec3[], flavor: ProjectionFlavor, params: ProjectorParams) {
    this.faces = sanitizeFaces(facesArg, x0.length);
    this.flavor = flavor;
    this.baseline = cloneVec3List(x0);
    this.positions = cloneVec3List(x0);
    this.params = { ...params };
    this.memory = createMemory(this.faces, this.positions);
  }

  reset(x0: Vec3[]): void {
    this.baseline = cloneVec3List(x0);
    this.positions = cloneVec3List(x0);
    this.handles.clear();
    this.memory = createMemory(this.faces, this.positions);
  }

  setBaseline(x0: Vec3[]): void {
    this.baseline = cloneVec3List(x0);
  }

  setHandles(handles: HandleSet): void {
    this.handles = new Map<number, Vec3>([...handles.targets.entries()].map(([k, v]) => [k, cloneVec3(v)]));
  }

  setParams(next: Partial<ProjectorParams>): void {
    this.params = { ...this.params, ...next };
  }

  step(iterations: number): void {
    if (iterations <= 0) return;
    const rho = Math.max(MIN_RHO_LEGACY, this.params.rho);
    const wFree = Math.max(0, this.params.wFree);
    const wHandle = Math.max(0, this.params.wHandle);
    const convexPasses = this.flavor === "convex" ? 1 : 0;
    const convexEps = CONVEX_HALFSPACE_EPS;
    const stepCap = LEGACY_STEP_CAP_RATIO * bboxScale(this.positions);

    for (let it = 0; it < iterations; it++) {
      for (let vi = 0; vi < this.positions.length; vi++) {
        const target = this.handles.get(vi) ?? this.baseline[vi];
        const w = this.handles.has(vi) ? wHandle : wFree;
        const deg = this.memory.incidence[vi].length;
        let sum: Vec3 = [0, 0, 0];
        for (let k = 0; k < deg; k++) {
          const { fi, li } = this.memory.incidence[vi][k];
          sum = v3.add(sum, v3.sub(this.memory.y[fi][li], this.memory.u[fi][li]));
        }
        let denom = w + rho * deg;
        if (this.flavor === "convex") {
          sum = v3.add(sum, v3.sub(this.memory.z[vi], this.memory.q[vi]));
          denom += rho;
        }
        const inv = 1 / Math.max(INVERSE_DENOM_EPS_LOOSE, denom);
        let next = v3.mul(v3.add(v3.mul(target, w), v3.mul(sum, rho)), inv);

        if (!Number.isFinite(next[0]) || !Number.isFinite(next[1]) || !Number.isFinite(next[2])) next = this.positions[vi];

        const delta = v3.sub(next, this.positions[vi]);
        const len = v3.norm(delta);
        if (len > stepCap) next = v3.add(this.positions[vi], v3.mul(delta, stepCap / Math.max(EPS, len)));
        this.positions[vi] = next;
      }

      updatePlanarYBlock(
        this.faces,
        this.positions,
        this.memory.u,
        this.memory.vbuf,
        this.memory.y,
        this.memory.prevFaceNormals,
        this.memory.facePlanes
      );

      if (convexPasses > 0) {
        const seed = this.positions.map((p, i) => v3.add(p, this.memory.q[i]));
        projectConvexHalfspaces(this.faces, this.memory, seed, convexPasses, convexEps);
      }

      updatePlanarDualBlock(this.faces, this.positions, this.memory.y, this.memory.u);

      if (convexPasses > 0) {
        for (let i = 0; i < this.positions.length; i++) {
          this.memory.q[i] = v3.add(this.memory.q[i], v3.sub(this.positions[i], this.memory.z[i]));
        }
      }

      this.memory.totalPlanarityViolation = computePlanarityViolationFromPlanes(
        this.faces,
        this.positions,
        this.memory.facePlanes
      );
    }
  }

  getPositionsRef(): ReadonlyArray<Vec3> {
    return this.positions;
  }

  snapshotPositions(): Vec3[] {
    return cloneVec3List(this.positions);
  }

  diagnostics(): { totalPlanarityViolation: number } {
    return { totalPlanarityViolation: this.memory.totalPlanarityViolation };
  }
}
