import type React from "react";
import type * as THREE from "three";
import type { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Vec3 } from "../../engine/math/types";
import type { ProjectorParams } from "../../engine/projection";
import type { PolyDerivedCache } from "../../engine/poly";

export type ProjectionControllerAPI = {
  setHandle: (vid: number, p: Vec3) => void;
  clearHandle: (vid: number) => void;
  clearAllHandles: () => void;
  hasHandle: (vid: number) => boolean;
  getHandleTargets: () => ReadonlyMap<number, Vec3>;
  getHandleCount: () => number;
  getParams: () => ProjectorParams;
  getBaselineSnapshot: () => Vec3[];
  resetToBaseline: () => void;

  step: (iters: number) => void;

  getXRef: () => ReadonlyArray<Vec3>;
  getDerivedCache: () => PolyDerivedCache;
  snapshot: () => Vec3[];
  commitBaseline: (snap: Vec3[]) => void;

  diagnostics: () => { totalPlanarityViolation: number };
};

export type OverlayOptions = {
  showNormals: boolean;
  showCom: boolean;
  showProjections: boolean;
  showStability: boolean;
  showBasins: boolean;
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
  faces: number[][];

  setMouseFromEvent: (e: PointerEvent) => void;
  computeFaceNormalAndPoint: (
    fi: number,
    baseline: ReadonlyArray<Vec3>
  ) => { normal: THREE.Vector3; point: THREE.Vector3 } | null;

  syncSceneFromX: (X: ReadonlyArray<Vec3>) => void;
  setDerivedOverlay: (cache: PolyDerivedCache | null, options: OverlayOptions) => void;
  updateSpheresMaterial: (handles: ReadonlyMap<number, Vec3>) => void;
  zoomBy: (factor: number) => void;
  resetView: () => void;
  setAxesVisible: (visible: boolean) => void;
  setGridVisible: (visible: boolean) => void;

  dispose: () => void;
};
