import type React from "react";
import type * as THREE from "three";
import type { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Vec3 } from "../../engine/math/types";
import type { IProjector, ProjectorParams } from "../../engine/projection";

export type ProjectionControllerAPI = {
  projectorRef: React.MutableRefObject<IProjector | null>;
  paramsRef: React.MutableRefObject<ProjectorParams>;
  handlesRef: React.MutableRefObject<Map<number, Vec3>>;
  baselineRef: React.MutableRefObject<Vec3[]>;

  setHandle: (vid: number, p: Vec3) => void;
  clearHandle: (vid: number) => void;
  clearAllHandles: () => void;
  getHandleCount: () => number;

  step: (iters: number) => void;
  stepUntilTol: (maxIters: number, tol: number) => void;

  getXRef: () => ReadonlyArray<Vec3>;
  snapshot: () => Vec3[];
  commitBaseline: (snap: Vec3[]) => void;

  diagnostics: () => { totalPlanarityViolation: number };
};

export type ThreeSceneAPI = {
  mountRef: React.RefObject<HTMLDivElement | null>;

  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbit: OrbitControls;
  raycaster: THREE.Raycaster;
  mouseNDC: THREE.Vector2;

  vMeshes: THREE.Mesh[];
  mesh: THREE.Mesh;
  triToFace: number[];

  setMouseFromEvent: (e: PointerEvent) => void;
  computeFaceNormalAndPoint: (
    fi: number,
    baseline: ReadonlyArray<Vec3>
  ) => { normal: THREE.Vector3; point: THREE.Vector3 } | null;

  syncSceneFromX: (X: ReadonlyArray<Vec3>) => void;
  updateSpheresMaterial: (handles: ReadonlyMap<number, Vec3>) => void;
  zoomBy: (factor: number) => void;
  resetView: () => void;
  setAxesVisible: (visible: boolean) => void;

  dispose: () => void;
};
