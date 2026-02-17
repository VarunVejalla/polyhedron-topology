import React, { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Vec3 } from "../../engine/math/types";
import type { PolyDerivedCache } from "../../engine/poly";
import type { OverlayOptions, ThreeSceneAPI } from "./types";

function makeLabelSprite(text: string, color: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    const fallback = new THREE.Sprite(new THREE.SpriteMaterial({ color }));
    fallback.scale.setScalar(0.12);
    return fallback;
  }

  canvas.width = 256;
  canvas.height = 96;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "700 50px Trebuchet MS, Segoe UI, Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineWidth = 6;
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.strokeText(text, canvas.width / 2, canvas.height / 2);
  ctx.fillStyle = color;
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.renderOrder = 10;
  return sprite;
}

function clearGroup(group: THREE.Group) {
  for (const obj of group.children) {
    const sprite = obj as THREE.Sprite;
    const mat = sprite.material as THREE.SpriteMaterial | undefined;
    if (mat?.map) mat.map.dispose();
    mat?.dispose();
  }
  group.clear();
}

function disposeObjectDeep(obj: THREE.Object3D) {
  obj.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
    const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(mat)) {
      for (const m of mat) m.dispose();
    } else {
      mat?.dispose();
    }
  });
}

const basinPalette = [
  new THREE.Color("#456990"),
  new THREE.Color("#f45b69"),
  new THREE.Color("#6ecb63"),
  new THREE.Color("#f6ae2d"),
  new THREE.Color("#7d5ba6"),
];

function zoomStepFromDistance(dist: number): number {
  const target = Math.max(0.05, dist / 7);
  const pow = Math.pow(10, Math.floor(Math.log10(target)));
  const candidates = [1, 2, 5].map((v) => v * pow);
  let best = candidates[0];
  let bestErr = Math.abs(target - best);
  for (const c of candidates) {
    const err = Math.abs(target - c);
    if (err < bestErr) {
      bestErr = err;
      best = c;
    }
  }
  return best;
}

export function useThreePolyhedronScene(
  mountRef: React.RefObject<HTMLDivElement | null>,
  faces: number[][],
  initialVertices: Vec3[]
): ThreeSceneAPI | null {
  const topologyKey = useMemo(() => JSON.stringify(faces), [faces]);
  const initialVerticesRef = React.useRef<Vec3[]>(initialVertices.map((p) => [p[0], p[1], p[2]] as Vec3));
  const [api, setApi] = useState<ThreeSceneAPI | null>(null);

  useEffect(() => {
    initialVerticesRef.current = initialVertices.map((p) => [p[0], p[1], p[2]] as Vec3);
  }, [initialVertices]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const X0 = initialVerticesRef.current;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf6f6f6);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 2000);
    const defaultCameraPos = new THREE.Vector3(2.5, 2.0, 2.5);
    const defaultTarget = new THREE.Vector3(0, 0, 0);
    camera.position.copy(defaultCameraPos);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    mount.appendChild(renderer.domElement);

    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.08;
    orbit.target.copy(defaultTarget);

    const axes = new THREE.AxesHelper(1.25);
    axes.visible = false;
    scene.add(axes);

    const grid = new THREE.GridHelper(10, 20, 0x8892a0, 0xc7cfd9);
    grid.position.y = -0.001;
    grid.visible = false;
    scene.add(grid);

    const axisLabels = new THREE.Group();
    axisLabels.visible = false;
    scene.add(axisLabels);

    const analysisRoot = new THREE.Group();
    scene.add(analysisRoot);
    const normalsGroup = new THREE.Group();
    const comGroup = new THREE.Group();
    const projectionsGroup = new THREE.Group();
    analysisRoot.add(normalsGroup, comGroup, projectionsGroup);

    let currentOverlayOptions: OverlayOptions = {
      showNormals: false,
      showCom: false,
      showProjections: false,
      showStability: false,
      showBasins: false,
    };
    let currentDerived: PolyDerivedCache | null = null;

    let labelStep = Number.NaN;
    const rebuildLabels = (step: number) => {
      clearGroup(axisLabels);
      labelStep = step;

      const extent = step * 5;
      for (let v = -extent; v <= extent + step * 0.5; v += step) {
        if (Math.abs(v) < step * 0.25) continue;
        const text = Number(v.toFixed(6)).toString();

        const xLabel = makeLabelSprite(text, "#cc3526");
        xLabel.position.set(v, 0.02, 0);
        axisLabels.add(xLabel);

        const zLabel = makeLabelSprite(text, "#2b73d2");
        zLabel.position.set(0, 0.02, v);
        axisLabels.add(zLabel);

        const yLabel = makeLabelSprite(text, "#2f9e44");
        yLabel.position.set(0.04, v, 0);
        axisLabels.add(yLabel);
      }
    };

    const updateLabelScaleAndDensity = () => {
      if (!grid.visible) return;
      const dist = camera.position.distanceTo(orbit.target);
      const step = zoomStepFromDistance(dist);
      if (!Number.isFinite(labelStep) || Math.abs(step - labelStep) > 1e-12) {
        rebuildLabels(step);
      }
      const s = Math.max(0.08, dist * 0.03);
      for (const child of axisLabels.children) {
        (child as THREE.Sprite).scale.set(s * 1.2, s * 0.6, 1);
      }
    };

    // --- Triangulate polygon faces (fan triangulation)
    const triangles: number[][] = [];
    const triToFace: number[] = [];
    for (let fi = 0; fi < faces.length; fi++) {
      const f = faces[fi];
      if (f.length < 3) continue;
      for (let i = 1; i + 1 < f.length; i++) {
        triangles.push([f[0], f[i], f[i + 1]]);
        triToFace.push(fi);
      }
    }

    const geom = new THREE.BufferGeometry();
    const triPos = new Float32Array(triangles.length * 3 * 3);
    const triCol = new Float32Array(triangles.length * 3 * 3);

    const writeTriPositions = (X: ReadonlyArray<Vec3>) => {
      let dst = 0;
      for (let ti = 0; ti < triangles.length; ti++) {
        const [a, b, c] = triangles[ti];
        const pa = X[a];
        const pb = X[b];
        const pc = X[c];
        if (!pa || !pb || !pc) {
          triPos[dst++] = 0;
          triPos[dst++] = 0;
          triPos[dst++] = 0;
          triPos[dst++] = 0;
          triPos[dst++] = 0;
          triPos[dst++] = 0;
          triPos[dst++] = 0;
          triPos[dst++] = 0;
          triPos[dst++] = 0;
          continue;
        }
        triPos[dst++] = pa[0];
        triPos[dst++] = pa[1];
        triPos[dst++] = pa[2];
        triPos[dst++] = pb[0];
        triPos[dst++] = pb[1];
        triPos[dst++] = pb[2];
        triPos[dst++] = pc[0];
        triPos[dst++] = pc[1];
        triPos[dst++] = pc[2];
      }
    };

    const writeTriColors = (derived: PolyDerivedCache | null, options: OverlayOptions) => {
      let dst = 0;
      for (let ti = 0; ti < triangles.length; ti++) {
        const fi = triToFace[ti] ?? 0;
        let color: [number, number, number] = [0.7, 0.7, 0.7];
        if (derived && options.showBasins) {
          const bi = derived.basinColorByFace[fi] ?? 0;
          const c = basinPalette[bi % basinPalette.length] ?? new THREE.Color(0.7, 0.7, 0.7);
          color = [c.r, c.g, c.b];
        } else if (derived) {
          const ci = derived.defaultColorByFace[fi] ?? (fi % basinPalette.length);
          const c = basinPalette[ci % basinPalette.length] ?? new THREE.Color(0.7, 0.7, 0.7);
          color = [c.r, c.g, c.b];
        } else {
          const c = basinPalette[fi % basinPalette.length] ?? new THREE.Color(0.7, 0.7, 0.7);
          color = [c.r, c.g, c.b];
        }

        if (derived && options.showStability) {
          const stable = derived.stableFace[fi] ?? false;
          color = stable ? [0.3, 0.75, 0.35] : [0.86, 0.38, 0.34];
        }
        const [r, g, b] = color;
        for (let k = 0; k < 3; k++) {
          triCol[dst++] = r;
          triCol[dst++] = g;
          triCol[dst++] = b;
        }
      }
    };

    writeTriPositions(X0);
    writeTriColors(null, currentOverlayOptions);

    geom.setAttribute("position", new THREE.BufferAttribute(triPos, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(triCol, 3));

    const mat = new THREE.MeshBasicMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9,
    });

    const mesh = new THREE.Mesh(geom, mat);
    scene.add(mesh);

    const edgeSet = new Set<string>();
    const edges: Array<[number, number]> = [];
    for (const cyc of faces) {
      for (let i = 0; i < cyc.length; i++) {
        const a = cyc[i];
        const b = cyc[(i + 1) % cyc.length];
        const u = Math.min(a, b);
        const v = Math.max(a, b);
        const key = `${u},${v}`;
        if (!edgeSet.has(key)) {
          edgeSet.add(key);
          edges.push([u, v]);
        }
      }
    }

    const edgeGeom = new THREE.BufferGeometry();
    const edgePos = new Float32Array(edges.length * 2 * 3);

    const writeEdgePositions = (X: ReadonlyArray<Vec3>) => {
      let dst = 0;
      for (const [u, v] of edges) {
        const pu = X[u];
        const pv = X[v];
        if (!pu || !pv) {
          edgePos[dst++] = 0;
          edgePos[dst++] = 0;
          edgePos[dst++] = 0;
          edgePos[dst++] = 0;
          edgePos[dst++] = 0;
          edgePos[dst++] = 0;
          continue;
        }
        edgePos[dst++] = pu[0];
        edgePos[dst++] = pu[1];
        edgePos[dst++] = pu[2];
        edgePos[dst++] = pv[0];
        edgePos[dst++] = pv[1];
        edgePos[dst++] = pv[2];
      }
    };

    writeEdgePositions(X0);
    edgeGeom.setAttribute("position", new THREE.BufferAttribute(edgePos, 3));

    const edgeMat = new THREE.LineBasicMaterial({ color: 0x111111 });
    const edgeLines = new THREE.LineSegments(edgeGeom, edgeMat);
    scene.add(edgeLines);

    const n = X0.length;
    const vMeshes: THREE.Mesh[] = [];
    const sphereGeom = new THREE.SphereGeometry(0.04, 14, 10);
    const matFree = new THREE.MeshBasicMaterial({ color: 0x444444 });
    const matHandle = new THREE.MeshBasicMaterial({ color: 0xff3344 });

    for (let i = 0; i < n; i++) {
      const m = new THREE.Mesh(sphereGeom, matFree);
      m.userData.vertexIndex = i;
      m.position.set(X0[i][0], X0[i][1], X0[i][2]);
      vMeshes.push(m);
      scene.add(m);
    }

    const updateSpheresMaterial = (handles: ReadonlyMap<number, Vec3>) => {
      for (const vm of vMeshes) {
        const vid = vm.userData.vertexIndex as number;
        vm.material = handles.has(vid) ? matHandle : matFree;
      }
    };

    const raycaster = new THREE.Raycaster();
    const mouseNDC = new THREE.Vector2();

    const setMouseFromEvent = (e: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;
      mouseNDC.set(2 * x - 1, 1 - 2 * y);
    };

    const computeFaceNormalAndPoint = (fi: number, baseline: ReadonlyArray<Vec3>): { normal: THREE.Vector3; point: THREE.Vector3 } | null => {
      if (fi < 0 || fi >= faces.length) return null;
      const cyc = faces[fi];
      if (cyc.length < 3) return null;
      const nrm = new THREE.Vector3(0, 0, 0);
      const cen = new THREE.Vector3(0, 0, 0);
      for (const vi of cyc) {
        const p = baseline[vi];
        cen.add(new THREE.Vector3(p[0], p[1], p[2]));
      }
      cen.multiplyScalar(1 / cyc.length);
      for (let i = 0; i < cyc.length; i++) {
        const a = baseline[cyc[i]];
        const b = baseline[cyc[(i + 1) % cyc.length]];
        nrm.x += (a[1] - b[1]) * (a[2] + b[2]);
        nrm.y += (a[2] - b[2]) * (a[0] + b[0]);
        nrm.z += (a[0] - b[0]) * (a[1] + b[1]);
      }
      if (nrm.lengthSq() < 1e-18) {
        const p0 = new THREE.Vector3(...baseline[cyc[0]]);
        const p1 = new THREE.Vector3(...baseline[cyc[1]]);
        const p2 = new THREE.Vector3(...baseline[cyc[2]]);
        nrm.copy(p1.clone().sub(p0).cross(p2.clone().sub(p0)));
      }
      if (nrm.lengthSq() < 1e-18) return null;
      nrm.normalize();
      return { normal: nrm, point: cen };
    };

    const clearOverlayGroup = (group: THREE.Group) => {
      for (const child of [...group.children]) {
        disposeObjectDeep(child);
        group.remove(child);
      }
    };

    const rebuildAnalysisOverlay = () => {
      writeTriColors(currentDerived, currentOverlayOptions);
      (geom.getAttribute("color") as THREE.BufferAttribute).needsUpdate = true;

      clearOverlayGroup(normalsGroup);
      clearOverlayGroup(comGroup);
      clearOverlayGroup(projectionsGroup);

      normalsGroup.visible = currentOverlayOptions.showNormals;
      comGroup.visible = currentOverlayOptions.showCom;
      projectionsGroup.visible = currentOverlayOptions.showProjections;

      if (!currentDerived) return;

      if (currentOverlayOptions.showCom) {
        const g = new THREE.SphereGeometry(0.05, 14, 10);
        const m = new THREE.MeshBasicMaterial({ color: 0xfee440 });
        const s = new THREE.Mesh(g, m);
        const c = currentDerived.centerOfMass;
        s.position.set(c[0], c[1], c[2]);
        comGroup.add(s);
      }

      if (currentOverlayOptions.showProjections) {
        for (let fi = 0; fi < faces.length; fi++) {
          const p = currentDerived.projectedComByFace[fi];
          const stable = currentDerived.stableFace[fi] ?? false;
          const g = new THREE.SphereGeometry(0.03, 10, 8);
          const m = new THREE.MeshBasicMaterial({ color: stable ? 0x24b06f : 0xd64f45 });
          const s = new THREE.Mesh(g, m);
          s.position.set(p[0], p[1], p[2]);
          projectionsGroup.add(s);
        }
      }

      if (currentOverlayOptions.showNormals) {
        for (let fi = 0; fi < faces.length; fi++) {
          const n = currentDerived.faceNormals[fi];
          const c = currentDerived.faceCentroids[fi];
          if (!n || !c) continue;
          const len = Math.hypot(n[0], n[1], n[2]);
          if (len <= 1e-12) continue;

          const dir = new THREE.Vector3(n[0] / len, n[1] / len, n[2] / len);
          const origin = new THREE.Vector3(c[0], c[1], c[2]);
          const length = 0.28;
          const headLength = 0.08;
          const headWidth = 0.045;
          const colorIndex = currentOverlayOptions.showBasins
            ? (currentDerived.basinColorByFace[fi] ?? (fi % basinPalette.length))
            : (currentDerived.defaultColorByFace[fi] ?? (fi % basinPalette.length));
          const color = basinPalette[colorIndex % basinPalette.length] ?? new THREE.Color(0.7, 0.7, 0.7);
          const arrow = new THREE.ArrowHelper(dir, origin, length, color.getHex(), headLength, headWidth);
          normalsGroup.add(arrow);
        }
      }
    };

    const syncSceneFromX = (X: ReadonlyArray<Vec3>) => {
      if (X.length !== n) return;
      writeTriPositions(X);
      (geom.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
      writeEdgePositions(X);
      (edgeGeom.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
      for (let i = 0; i < n; i++) vMeshes[i].position.set(X[i][0], X[i][1], X[i][2]);
      rebuildAnalysisOverlay();
    };

    const resize = () => {
      const w = Math.max(1, mount.clientWidth);
      const h = Math.max(1, mount.clientHeight);
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(mount);

    let raf = 0;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      orbit.update();
      updateLabelScaleAndDensity();
      renderer.render(scene, camera);
    };
    animate();

    const dispose = () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      orbit.dispose();
      geom.dispose();
      edgeGeom.dispose();
      sphereGeom.dispose();
      grid.geometry.dispose();
      if (Array.isArray(grid.material)) {
        for (const m of grid.material) m.dispose();
      } else {
        grid.material.dispose();
      }
      axes.geometry.dispose();
      if (Array.isArray(axes.material)) {
        for (const m of axes.material) m.dispose();
      } else {
        axes.material.dispose();
      }
      clearGroup(axisLabels);
      clearOverlayGroup(normalsGroup);
      clearOverlayGroup(comGroup);
      clearOverlayGroup(projectionsGroup);
      scene.remove(analysisRoot);
      (mesh.material as THREE.Material).dispose();
      edgeMat.dispose();
      matFree.dispose();
      matHandle.dispose();
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
    };

    const sceneApi: ThreeSceneAPI = {
      mountRef,
      renderer,
      camera,
      orbit,
      raycaster,
      mouseNDC,
      vMeshes,
      mesh,
      triToFace,
      setMouseFromEvent,
      computeFaceNormalAndPoint,
      syncSceneFromX,
      setDerivedOverlay: (cache: PolyDerivedCache | null, options: OverlayOptions) => {
        currentDerived = cache;
        currentOverlayOptions = { ...options };
        rebuildAnalysisOverlay();
      },
      updateSpheresMaterial,
      zoomBy: (factor: number) => {
        if (!Number.isFinite(factor) || factor <= 0) return;
        const toCamera = camera.position.clone().sub(orbit.target);
        const dist = toCamera.length();
        if (dist <= 1e-9) return;
        const nextDist = THREE.MathUtils.clamp(dist * factor, 0.2, 120);
        toCamera.setLength(nextDist);
        camera.position.copy(orbit.target.clone().add(toCamera));
        camera.updateProjectionMatrix();
        orbit.update();
      },
      resetView: () => {
        camera.position.copy(defaultCameraPos);
        orbit.target.copy(defaultTarget);
        camera.updateProjectionMatrix();
        orbit.update();
      },
      setAxesVisible: (visible: boolean) => {
        axes.visible = visible;
      },
      setGridVisible: (visible: boolean) => {
        grid.visible = visible;
        axisLabels.visible = visible;
        if (!visible) {
          clearGroup(axisLabels);
          labelStep = Number.NaN;
          return;
        }
        const dist = camera.position.distanceTo(orbit.target);
        rebuildLabels(zoomStepFromDistance(dist));
        updateLabelScaleAndDensity();
      },
      dispose,
    };

    setApi(sceneApi);

    return () => {
      setApi(null);
      dispose();
    };
  }, [topologyKey, mountRef, faces]);

  useEffect(() => {
    if (!api) return;
    api.syncSceneFromX(initialVertices);
  }, [api, initialVertices]);

  return api;
}
