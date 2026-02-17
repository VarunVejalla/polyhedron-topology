import { EPS } from "./constants";
import type { Vec3 } from "./types";

/**
 * Jacobi eigen-decomposition for a real symmetric 3x3 matrix.
 * Returns eigenvalues and eigenvectors (columns of V).
 * This is robust enough for interactive geometry (small meshes).
 */
function jacobiEigenSym3(Ain: number[][], iters = 30): { values: Vec3; vectors: number[][] } {
  const A = [
    [Ain[0][0], Ain[0][1], Ain[0][2]],
    [Ain[1][0], Ain[1][1], Ain[1][2]],
    [Ain[2][0], Ain[2][1], Ain[2][2]],
  ];
  const V = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];

  const abs = Math.abs;
  for (let k = 0; k < iters; k++) {
    let p = 0, q = 1;
    let max = abs(A[0][1]);
    if (abs(A[0][2]) > max) { p = 0; q = 2; max = abs(A[0][2]); }
    if (abs(A[1][2]) > max) { p = 1; q = 2; max = abs(A[1][2]); }
    if (max < EPS) break;

    const app = A[p][p];
    const aqq = A[q][q];
    const apq = A[p][q];
    const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(phi);
    const s = Math.sin(phi);

    // Left-multiply rows r,p/q
    for (let r = 0; r < 3; r++) {
      const arp = A[r][p];
      const arq = A[r][q];
      A[r][p] = c * arp - s * arq;
      A[r][q] = s * arp + c * arq;
    }
    // Right-multiply cols p/q
    for (let r = 0; r < 3; r++) {
      const apr = A[p][r];
      const aqr = A[q][r];
      A[p][r] = c * apr - s * aqr;
      A[q][r] = s * apr + c * aqr;
    }

    A[p][q] = 0;
    A[q][p] = 0;

    // Update V = V R
    for (let r = 0; r < 3; r++) {
      const vrp = V[r][p];
      const vrq = V[r][q];
      V[r][p] = c * vrp - s * vrq;
      V[r][q] = s * vrp + c * vrq;
    }
  }

  const values: Vec3 = [A[0][0], A[1][1], A[2][2]];
  return { values, vectors: V };
}

export function smallestEigenvectorSym3(cov: number[][]): Vec3 {
  const { values, vectors } = jacobiEigenSym3(cov);
  let idx = 0;
  if (values[1] < values[idx]) idx = 1;
  if (values[2] < values[idx]) idx = 2;
  return [vectors[0][idx], vectors[1][idx], vectors[2][idx]];
}
