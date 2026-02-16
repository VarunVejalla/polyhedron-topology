import type { SimpleGraph } from "../graph/types";
import { cloneGraph } from "../graph/core";
import { buildVertexPresetGraph, circleLayout, presetNames } from "../graph/presets";
import { facesFromEmbedding, chooseOuterFace, planarDualFromFaces, type Face } from "../graph/embedding";
import { tutteLayout } from "../graph/layout";
import { makeDefaultPrism, PRISM_FACES, type Vec3 } from "../engine/prismTopology";
import { checkPolyhedral } from "../graph/validity";
import type { ProjectionMethod } from "../engine/projection";

export const GRAPH_VIEW = { w: 420, h: 360, padding: 28 };

type PolyDocument = {
  vertices: Vec3[];
  faces: number[][];
};

type UIState = {
  leftWidth: number;
  showGraphs: boolean;
  show3D: boolean;
  showAxes: boolean;
  showGrid: boolean;
  showVertexPositions: boolean;
  showNormals: boolean;
  showCom: boolean;
  showProjections: boolean;
  showStability: boolean;
  showBasins: boolean;
  showAdvancedSettings: boolean;
};

type ProjectionState = {
  method: ProjectionMethod;
  rho: number;
  wFree: number;
  wHandle: number;
  lambdaReg: number;
  itersPerFrame: number;
  itersOnRelease: number;
  hardProjectMode: "iters" | "tol";
  hardProjectMaxIters: number;
  hardProjectTolPlanar: number;
};

type Document = {
  preset: string;
  vertexGraph: SimpleGraph;
  faceGraph: SimpleGraph;
  poly: PolyDocument;
  projection: ProjectionState;
  ui: UIState;
};

type DocumentState = {
  present: Document;
  past: Document[];
  future: Document[];
};

type DocumentAction =
  | { type: "UNDO" }
  | { type: "REDO" }

  // ---- High-level mutations (these are the ONLY ones that touch history)
  | { type: "APPLY_PRESET"; preset: string }
  | { type: "COMMIT_VERTEX_GRAPH"; graph: SimpleGraph }
  | { type: "COMMIT_FACE_GRAPH"; graph: SimpleGraph }
  | { type: "COMMIT_POLY"; poly: PolyDocument }
  | { type: "COMMIT_BUILD"; patch: Partial<Pick<Document, "vertexGraph" | "faceGraph" | "poly">> }
  | { type: "COMMIT_POLY_VERTICES"; vertices: Vec3[] }

  // ---- Live (non-history) edits for dragging / interactive updates
  | { type: "LIVE_VERTEX_GRAPH"; graph: SimpleGraph }
  | { type: "LIVE_FACE_GRAPH"; graph: SimpleGraph }
  | { type: "LIVE_POLY_VERTICES"; vertices: Vec3[] }

  // ---- Pure parameter/UI edits (do not touch history by default)
  | { type: "SET_PROJECTION"; patch: Partial<ProjectionState> }
  | { type: "SET_UI"; patch: Partial<UIState> }
  | { type: "SET_PRESET_ONLY"; preset: string }; // rarely used; does not reset graphs


function cloneVerts(v: Vec3[]): Vec3[] {
  return v.map((p) => [...p] as Vec3);
}
function cloneFaces(f: number[][]): number[][] {
  return f.map((cy) => [...cy]);
}
function clonePoly(p: PolyDocument): PolyDocument {
  return { vertices: cloneVerts(p.vertices), faces: cloneFaces(p.faces) };
}
function cloneDoc(d: Document): Document {
  return {
    preset: d.preset,
    vertexGraph: cloneGraph(d.vertexGraph),
    faceGraph: cloneGraph(d.faceGraph),
    poly: clonePoly(d.poly),
    projection: { ...d.projection },
    ui: { ...d.ui },
  };
}

function computeDualPairFromVertexGraph(g0: SimpleGraph): { vertexGraph: SimpleGraph; faceGraph: SimpleGraph } | null {
  const rep = checkPolyhedral(g0);
  if (!rep.ok) return null;

  let faces0: Face[];
  try {
    faces0 = facesFromEmbedding(g0.nodes.map((n) => n.id), g0.edges, rep.embedding);
  } catch {
    return null;
  }

  const outerFace = chooseOuterFace(faces0);
  if (!outerFace) return null;

  const posTutte = tutteLayout(g0, outerFace.cycle, GRAPH_VIEW);
  const vertexGraph: SimpleGraph = {
    nodes: g0.nodes.map((n) => {
      const p = posTutte.get(n.id);
      return p ? { ...n, x: p.x, y: p.y } : n;
    }),
    edges: g0.edges.map((e) => ({ ...e })),
  };

  let faceGraph = planarDualFromFaces(faces0);

  // Give the dual a pleasant initial layout (circle), preserving later manual edits.
  const r = Math.max(40, Math.min(GRAPH_VIEW.w, GRAPH_VIEW.h) / 2 - GRAPH_VIEW.padding);
  const posCircle = circleLayout(faceGraph.nodes.map((n) => n.id), GRAPH_VIEW.w, GRAPH_VIEW.h, r);
  faceGraph = {
    nodes: faceGraph.nodes.map((n) => ({ ...n, x: posCircle[n.id].x, y: posCircle[n.id].y })),
    edges: faceGraph.edges.map((e) => ({ ...e })),
  };

  return { vertexGraph, faceGraph };
}


export function createInitialState(): DocumentState {
  const initialPoly = makeDefaultPrism();

  const presets = presetNames();
  const preset = presets[0] ?? "Triangular prism";
  const initialVertexGraph = buildVertexPresetGraph(preset, GRAPH_VIEW);

  const pair = computeDualPairFromVertexGraph(initialVertexGraph);

  const present: Document = {
    preset,
    vertexGraph: pair?.vertexGraph ?? initialVertexGraph,
    faceGraph: pair?.faceGraph ?? initialVertexGraph,
    poly: { vertices: initialPoly.vertices, faces: PRISM_FACES },
    projection: {
      method: "admm",
      rho: 10,
      wFree: 1,
      wHandle: 1e5,
      lambdaReg: 0,
      itersPerFrame: 10,
      itersOnRelease: 120,
      hardProjectMode: "iters",
      hardProjectMaxIters: 400,
      hardProjectTolPlanar: 1e-6,
    },
    ui: {
      leftWidth: 460,
      showGraphs: true,
      show3D: true,
      showAxes: false,
      showGrid: false,
      showVertexPositions: false,
      showNormals: false,
      showCom: false,
      showProjections: false,
      showStability: false,
      showBasins: false,
      showAdvancedSettings: false,
    },
  };

  return { present, past: [], future: [] };
}

export function documentReducer(state: DocumentState, action: DocumentAction): DocumentState {
  const commit = (nextPresent: Document): DocumentState => ({
    present: cloneDoc(nextPresent),
    past: [...state.past, cloneDoc(state.present)],
    future: [],
  });

  // const live = (nextPresent: Document): DocumentState => ({
  //   ...state,
  //   present: cloneDoc(nextPresent),
  //   // Any new edit invalidates redo history.
  //   future: [],
  // });

  const liveUpdate = (updates: Partial<Document>, opts?: { clearFuture?: boolean }): DocumentState => ({
    ...state,
    present: { ...state.present, ...updates },
    future: opts?.clearFuture === false ? state.future : [],
  });

  switch (action.type) {
    case "UNDO": {
      if (state.past.length === 0) return state;
      const prev = state.past[state.past.length - 1];
      const newPast = state.past.slice(0, -1);
      return { present: cloneDoc(prev), past: newPast, future: [cloneDoc(state.present), ...state.future] };
    }

    case "REDO": {
      if (state.future.length === 0) return state;
      const next = state.future[0];
      const newFuture = state.future.slice(1);
      return { present: cloneDoc(next), past: [...state.past, cloneDoc(state.present)], future: newFuture };
    }

    case "APPLY_PRESET": {
      const initialVertexGraph = buildVertexPresetGraph(action.preset, GRAPH_VIEW);
      const pair = computeDualPairFromVertexGraph(initialVertexGraph);
      const next: Document = {
        ...cloneDoc(state.present),
        preset: action.preset,
        vertexGraph: cloneGraph(pair?.vertexGraph ?? initialVertexGraph),
        faceGraph: cloneGraph(pair?.faceGraph ?? initialVertexGraph),
        // poly is intentionally NOT changed; user must explicitly build canonical.
      };
      return commit(next);
    }

    case "COMMIT_VERTEX_GRAPH": {
      const next: Document = { ...cloneDoc(state.present), vertexGraph: cloneGraph(action.graph) };
      return commit(next);
    }

    case "COMMIT_FACE_GRAPH": {
      const next: Document = { ...cloneDoc(state.present), faceGraph: cloneGraph(action.graph) };
      return commit(next);
    }

    case "COMMIT_POLY": {
      const next: Document = { ...cloneDoc(state.present), poly: clonePoly(action.poly) };
      return commit(next);
    }

    case "COMMIT_BUILD": {
      const next: Document = {
        ...cloneDoc(state.present),
        ...(action.patch.vertexGraph ? { vertexGraph: cloneGraph(action.patch.vertexGraph) } : {}),
        ...(action.patch.faceGraph ? { faceGraph: cloneGraph(action.patch.faceGraph) } : {}),
        ...(action.patch.poly ? { poly: clonePoly(action.patch.poly) } : {}),
      };
      return commit(next);
    }

    case "COMMIT_POLY_VERTICES": {
      const next: Document = { ...cloneDoc(state.present), poly: { ...state.present.poly, vertices: cloneVerts(action.vertices) } };
      return commit(next);
    }

    // For LIVE graph/poly updates, clone only what changes:
    case "LIVE_VERTEX_GRAPH": {
      return liveUpdate({ vertexGraph: cloneGraph(action.graph) });
    }

    case "LIVE_FACE_GRAPH": {
      return liveUpdate({ faceGraph: cloneGraph(action.graph) });
    }

    case "LIVE_POLY_VERTICES": {
      return liveUpdate({ poly: { ...state.present.poly, vertices: cloneVerts(action.vertices) } });
    }

    case "SET_PROJECTION": {
      return liveUpdate({ projection: { ...state.present.projection, ...action.patch } }, { clearFuture: false });
    }

    case "SET_UI": {
      return liveUpdate({ ui: { ...state.present.ui, ...action.patch } }, { clearFuture: false });
    }

    case "SET_PRESET_ONLY": {
      return liveUpdate({ preset: action.preset }, { clearFuture: false });
    }

    default:
      return state;
  }
}
