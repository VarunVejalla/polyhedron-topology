import type { Vec3 } from "../../math/types";

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function regularityValueAndGradient(
  faces: ReadonlyArray<ReadonlyArray<number>>,
  positions: ReadonlyArray<Vec3>,
  epsArea: number,
  gradAccum?: Vec3[]
): number {
  let total = 0;
  for (let fi = 0; fi < faces.length; fi++) {
    const face = faces[fi];
    const nSides = face.length;
    if (nSides < 3) continue;

    let perimeter = 0;
    for (let i = 0; i < nSides; i++) {
      const ia = face[i];
      const ib = face[(i + 1) % nSides];
      const ex = positions[ib][0] - positions[ia][0];
      const ey = positions[ib][1] - positions[ia][1];
      const ez = positions[ib][2] - positions[ia][2];
      perimeter += Math.hypot(ex, ey, ez);
    }

    const areaVec: Vec3 = [0, 0, 0];
    for (let i = 0; i < nSides; i++) {
      const ia = face[i];
      const ib = face[(i + 1) % nSides];
      const a = positions[ia];
      const b = positions[ib];
      areaVec[0] += 0.5 * (a[1] * b[2] - a[2] * b[1]);
      areaVec[1] += 0.5 * (a[2] * b[0] - a[0] * b[2]);
      areaVec[2] += 0.5 * (a[0] * b[1] - a[1] * b[0]);
    }

    const area = Math.sqrt(dot3(areaVec, areaVec) + epsArea * epsArea);
    const invArea = 1 / Math.max(1e-12, area);
    const nHat: Vec3 = [areaVec[0] * invArea, areaVec[1] * invArea, areaVec[2] * invArea];

    const cReg = 4 * nSides * Math.tan(Math.PI / nSides);
    const invDen = 1 / Math.max(1e-12, cReg * area);
    const reg = perimeter * perimeter * invDen - 1;
    const regSq = reg * reg;
    total += regSq;

    if (!gradAccum) continue;

    const regScale = 2 * reg;
    const coeffP = regScale * 2 * perimeter * invDen;
    const coeffA = regScale * (-(perimeter * perimeter) / Math.max(1e-12, cReg * area * area));
    for (let i = 0; i < nSides; i++) {
      const vi = face[i];
      const prev = face[(i - 1 + nSides) % nSides];
      const next = face[(i + 1) % nSides];

      const p = positions[vi];
      const pPrev = positions[prev];
      const pNext = positions[next];

      const ePrev: Vec3 = [p[0] - pPrev[0], p[1] - pPrev[1], p[2] - pPrev[2]];
      const eNext: Vec3 = [p[0] - pNext[0], p[1] - pNext[1], p[2] - pNext[2]];
      const lPrev = Math.max(1e-12, Math.hypot(ePrev[0], ePrev[1], ePrev[2]));
      const lNext = Math.max(1e-12, Math.hypot(eNext[0], eNext[1], eNext[2]));
      const dP: Vec3 = [
        ePrev[0] / lPrev + eNext[0] / lNext,
        ePrev[1] / lPrev + eNext[1] / lNext,
        ePrev[2] / lPrev + eNext[2] / lNext,
      ];

      const edgePN: Vec3 = [pNext[0] - pPrev[0], pNext[1] - pPrev[1], pNext[2] - pPrev[2]];
      const dA = cross3(edgePN, nHat);
      dA[0] *= 0.5;
      dA[1] *= 0.5;
      dA[2] *= 0.5;

      gradAccum[vi][0] += coeffP * dP[0] + coeffA * dA[0];
      gradAccum[vi][1] += coeffP * dP[1] + coeffA * dA[1];
      gradAccum[vi][2] += coeffP * dP[2] + coeffA * dA[2];
    }
  }
  return total;
}

type VertexTrackingObjectiveOptions = {
  baseline: ReadonlyArray<Vec3>;
  handles: ReadonlyMap<number, Vec3>;
  wFree: number;
  wHandle: number;
  lambdaReg: number;
  epsArea: number;
  faces: ReadonlyArray<ReadonlyArray<number>>;
};

export function evaluateVertexTrackingObjectiveAndGradient(
  y: ReadonlyArray<number>,
  options: VertexTrackingObjectiveOptions,
  gradOut?: number[]
): number {
  const n = options.baseline.length;
  const vDim = 3 * n;
  if (gradOut) {
    for (let i = 0; i < vDim; i++) gradOut[i] = 0;
  }

  const positions: Vec3[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const b = 3 * i;
    positions[i] = [y[b], y[b + 1], y[b + 2]];
  }

  let energy = 0;
  for (let i = 0; i < n; i++) {
    const b = 3 * i;
    const handleTarget = options.handles.get(i);
    const w = handleTarget ? options.wHandle : options.wFree;
    const target = handleTarget ?? options.baseline[i];
    const dx = y[b] - target[0];
    const dy = y[b + 1] - target[1];
    const dz = y[b + 2] - target[2];
    energy += w * (dx * dx + dy * dy + dz * dz);
    if (gradOut) {
      gradOut[b] += 2 * w * dx;
      gradOut[b + 1] += 2 * w * dy;
      gradOut[b + 2] += 2 * w * dz;
    }
  }

  if (options.lambdaReg > 0) {
    const gradReg = gradOut ? positions.map(() => [0, 0, 0] as Vec3) : undefined;
    const reg = regularityValueAndGradient(options.faces, positions, options.epsArea, gradReg);
    energy += options.lambdaReg * reg;
    if (gradOut && gradReg) {
      for (let i = 0; i < n; i++) {
        const b = 3 * i;
        gradOut[b] += options.lambdaReg * gradReg[i][0];
        gradOut[b + 1] += options.lambdaReg * gradReg[i][1];
        gradOut[b + 2] += options.lambdaReg * gradReg[i][2];
      }
    }
  }

  return energy;
}
