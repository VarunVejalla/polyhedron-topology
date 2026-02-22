import type { SimpleGraph } from "../graph/types";
import { cloneGraph } from "../graph/core";
import { buildVertexPresetGraph, presetNames } from "../graph/presets";
import { GRAPH_VIEW } from "../graph/view";
import { makeDefaultPrism, PRISM_FACES, type Vec3 } from "../engine/prismTopology";
import { computeSignedVolumeFromVerticesAndFaces } from "../engine/poly";
import { deriveDualPairFromVertexGraph } from "../graph/pipeline";
import { createDefaultProjectionSettings, type ProjectionSettings } from "./projectionSettings";

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
  showAdvancedProjectionParams: boolean;
  showGraphicalSettings: boolean;
};

type ProjectionState = ProjectionSettings;

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

function sameFaces(a: ReadonlyArray<ReadonlyArray<number>>, b: ReadonlyArray<ReadonlyArray<number>>): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].length !== b[i].length) return false;
    for (let j = 0; j < a[i].length; j++) if (a[i][j] !== b[i][j]) return false;
  }
  return true;
}

function withTopologyVolumeTarget(doc: Document, nextPoly: PolyDocument): Document {
  const nextProjection = { ...doc.projection };
  if (!sameFaces(doc.poly.faces, nextPoly.faces)) {
    nextProjection.goalVolume = computeSignedVolumeFromVerticesAndFaces(nextPoly.vertices, nextPoly.faces);
  }
  return {
    ...doc,
    poly: clonePoly(nextPoly),
    projection: nextProjection,
  };
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

export function createInitialState(): DocumentState {
  const initialPoly = makeDefaultPrism();
  const initialVolume = computeSignedVolumeFromVerticesAndFaces(initialPoly.vertices, PRISM_FACES);

  const presets = presetNames();
  const preset = presets[0] ?? "Triangular prism";
  const initialVertexGraph = buildVertexPresetGraph(preset, GRAPH_VIEW);

  const pair = deriveDualPairFromVertexGraph(initialVertexGraph, GRAPH_VIEW);

  const present: Document = {
    preset,
    vertexGraph: pair?.vertexGraph ?? initialVertexGraph,
    faceGraph: pair?.faceGraph ?? initialVertexGraph,
    poly: { vertices: initialPoly.vertices, faces: PRISM_FACES },
    projection: createDefaultProjectionSettings(initialVolume),
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
      showAdvancedProjectionParams: false,
      showGraphicalSettings: false,
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
      const pair = deriveDualPairFromVertexGraph(initialVertexGraph, GRAPH_VIEW);
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
      const next = withTopologyVolumeTarget(cloneDoc(state.present), action.poly);
      return commit(next);
    }

    case "COMMIT_BUILD": {
      const base = cloneDoc(state.present);
      const withPoly = action.patch.poly ? withTopologyVolumeTarget(base, action.patch.poly) : base;
      const next: Document = {
        ...withPoly,
        ...(action.patch.vertexGraph ? { vertexGraph: cloneGraph(action.patch.vertexGraph) } : {}),
        ...(action.patch.faceGraph ? { faceGraph: cloneGraph(action.patch.faceGraph) } : {}),
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
