import React, { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Vec3 } from "../../engine/math/types";
import type { ThreeSceneAPI } from "./types";

function hashColor(fi: number): number {
  // A few pleasant, distinct-ish pastels (cycled).
  const palette = [0x8ecae6, 0x219ebc, 0xffb703, 0xfb8500, 0xbde0fe, 0xcaffbf, 0xffc6ff, 0xdee2ff];
  return palette[fi % palette.length];
}

export function useThreePolyhedronScene(
  mountRef: React.RefObject<HTMLDivElement | null>,
  faces: number[][],
  initialVertices: Vec3[]
): ThreeSceneAPI | null {
  const topologyKey = useMemo(() => JSON.stringify(faces), [faces]);
  const [api, setApi] = useState<ThreeSceneAPI | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    
    const X0 = initialVertices;

    // --- Scene
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

    // --- Face colors per face index
    const faceRGB: Array<[number, number, number]> = faces.map((_f, fi) => {
      const hex = hashColor(fi);
      const r = ((hex >> 16) & 255) / 255;
      const g = ((hex >> 8) & 255) / 255;
      const b = (hex & 255) / 255;
      return [r, g, b];
    });

    // --- Triangle mesh geometry
    const geom = new THREE.BufferGeometry();
    const triPos = new Float32Array(triangles.length * 3 * 3);
    const triCol = new Float32Array(triangles.length * 3 * 3);

    const writeTriPositions = (X: ReadonlyArray<Vec3>) => {
      let dst = 0;
      for (let ti = 0; ti < triangles.length; ti++) {
        const [a, b, c] = triangles[ti];
        const pa = X[a], pb = X[b], pc = X[c];
        triPos[dst++] = pa[0]; triPos[dst++] = pa[1]; triPos[dst++] = pa[2];
        triPos[dst++] = pb[0]; triPos[dst++] = pb[1]; triPos[dst++] = pb[2];
        triPos[dst++] = pc[0]; triPos[dst++] = pc[1]; triPos[dst++] = pc[2];
      }
    };

    const writeTriColors = () => {
      let dst = 0;
      for (let ti = 0; ti < triangles.length; ti++) {
        const fi = triToFace[ti] ?? 0;
        const [r, g, b] = faceRGB[fi] ?? [0.7, 0.7, 0.7];
        // Same face color for the 3 vertices of this triangle.
        for (let k = 0; k < 3; k++) {
          triCol[dst++] = r; triCol[dst++] = g; triCol[dst++] = b;
        }
      }
    };

    writeTriPositions(X0);
    writeTriColors();

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

    // --- Unique edges from polygon cycles (for nice crisp outlines)
    const edgeSet = new Set<string>();
    const edges: Array<[number, number]> = [];
    for (const cyc of faces) {
      for (let i = 0; i < cyc.length; i++) {
        const a = cyc[i];
        const b = cyc[(i + 1) % cyc.length];
        const u = Math.min(a, b), v = Math.max(a, b);
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
        const pu = X[u], pv = X[v];
        edgePos[dst++] = pu[0]; edgePos[dst++] = pu[1]; edgePos[dst++] = pu[2];
        edgePos[dst++] = pv[0]; edgePos[dst++] = pv[1]; edgePos[dst++] = pv[2];
      }
    };

    writeEdgePositions(X0);
    edgeGeom.setAttribute("position", new THREE.BufferAttribute(edgePos, 3));

    const edgeMat = new THREE.LineBasicMaterial({ color: 0x111111 });
    const edgeLines = new THREE.LineSegments(edgeGeom, edgeMat);
    scene.add(edgeLines);

    // --- Vertex spheres for picking
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

    // --- Picking helpers
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
      // Newell's method for polygon normal (stable for non-tri faces)
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
        // fallback: use triangle normal of first 3 verts
        const p0 = new THREE.Vector3(...baseline[cyc[0]]);
        const p1 = new THREE.Vector3(...baseline[cyc[1]]);
        const p2 = new THREE.Vector3(...baseline[cyc[2]]);
        nrm.copy(p1.clone().sub(p0).cross(p2.clone().sub(p0)));
      }
      if (nrm.lengthSq() < 1e-18) return null;
      nrm.normalize();
      return { normal: nrm, point: cen };
    };

    const syncSceneFromX = (X: ReadonlyArray<Vec3>) => {
      writeTriPositions(X);
      (geom.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
      writeEdgePositions(X);
      (edgeGeom.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
      for (let i = 0; i < n; i++) vMeshes[i].position.set(X[i][0], X[i][1], X[i][2]);
    };

    // --- Resize / animate
    const resize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
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
      axes.geometry.dispose();
      if (Array.isArray(axes.material)) {
        for (const m of axes.material) m.dispose();
      } else {
        axes.material.dispose();
      }
      // dispose materials (clones)
      (mesh.material as THREE.Material).dispose();
      edgeMat.dispose();
      matFree.dispose();
      matHandle.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
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
      updateSpheresMaterial,
      zoomBy: (factor: number) => {
        if (!Number.isFinite(factor) || factor <= 0) return;
        const toCamera = camera.position.clone().sub(orbit.target);
        const dist = toCamera.length();
        if (dist <= 1e-9) return;
        const nextDist = THREE.MathUtils.clamp(dist * factor, 0.2, 80);
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
      dispose,
    };

    setApi(sceneApi);

    return () => {
      setApi(null);
      dispose();
    };
  }, [topologyKey, mountRef]);

  // Same-topology vertex updates should not recreate the scene/camera.
  useEffect(() => {
    if (!api) return;
    api.syncSceneFromX(initialVertices);
  }, [api, initialVertices]);

  return api;
}
