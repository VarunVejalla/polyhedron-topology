import { presetNames, buildVertexPresetGraph } from "../src/graph/presets";
import { derivePolyFromFaceGraph, derivePolyFromVertexGraph, type GraphView } from "../src/graph/pipeline";
import { PRISM_FACES, makeDefaultPrism } from "../src/engine/prismTopology";
import { buildPolyAuxState } from "../src/engine/poly/auxiliary";
import { buildPolyState } from "../src/engine/poly/state";
import { buildPolyTopology } from "../src/engine/poly/topology";
import type { Vec3 } from "../src/engine/math/types";

type Poly = { vertices: Vec3[]; faces: number[][] };

type OrientationReport = {
  badEdgeCounts: number;
  badEdgeSigns: number;
  posFaceDots: number;
  negFaceDots: number;
  zeroFaceDots: number;
  volume: number;
};

const GRAPH_VIEW: GraphView = { w: 420, h: 360, padding: 28 };

function fail(msg: string): never {
  throw new Error(msg);
}

function analyzeOrientation(poly: Poly): OrientationReport {
  const state = buildPolyState(poly.vertices, poly.faces);
  const topology = buildPolyTopology(poly.faces, poly.vertices.length);
  const aux = buildPolyAuxState(state, topology);

  const counts = new Array<number>(topology.edges.length).fill(0);
  const signSums = new Array<number>(topology.edges.length).fill(0);
  for (const inc of topology.edgeIncidencesFlat) {
    counts[inc.edgeIndex] += 1;
    signSums[inc.edgeIndex] += inc.sign;
  }

  let posFaceDots = 0;
  let negFaceDots = 0;
  let zeroFaceDots = 0;
  for (let fi = 0; fi < state.faces.length; fi++) {
    const face = state.faces[fi];
    let ax = 0;
    let ay = 0;
    let az = 0;
    for (let i = 0; i < face.length; i++) {
      const p = state.vertices[face[i]];
      const q = state.vertices[face[(i + 1) % face.length]];
      ax += 0.5 * (p[1] * q[2] - p[2] * q[1]);
      ay += 0.5 * (p[2] * q[0] - p[0] * q[2]);
      az += 0.5 * (p[0] * q[1] - p[1] * q[0]);
    }
    const n = state.facePlanes[fi].n;
    const dot = n[0] * ax + n[1] * ay + n[2] * az;
    if (dot > 1e-8) posFaceDots++;
    else if (dot < -1e-8) negFaceDots++;
    else zeroFaceDots++;
  }

  return {
    badEdgeCounts: counts.filter((c) => c !== 2).length,
    badEdgeSigns: signSums.filter((s) => s !== 0).length,
    posFaceDots,
    negFaceDots,
    zeroFaceDots,
    volume: aux.volume,
  };
}

function assertConvexOrientation(name: string, poly: Poly) {
  const r = analyzeOrientation(poly);
  if (r.badEdgeCounts !== 0) fail(`${name}: non-manifold edge valence count detected (${r.badEdgeCounts})`);
  if (r.badEdgeSigns !== 0) fail(`${name}: shared-edge incidence signs do not cancel (${r.badEdgeSigns})`);
  if (r.negFaceDots !== 0 || r.zeroFaceDots !== 0) {
    fail(`${name}: face winding vs outward normal mismatch (neg=${r.negFaceDots}, zero=${r.zeroFaceDots})`);
  }
  if (!(r.volume > 0)) fail(`${name}: non-positive volume (${r.volume})`);
}

function run() {
  const prism = makeDefaultPrism();
  assertConvexOrientation("default-prism", { vertices: prism.vertices, faces: PRISM_FACES });

  for (const preset of presetNames()) {
    const g = buildVertexPresetGraph(preset, GRAPH_VIEW);
    const fromVertex = derivePolyFromVertexGraph(g, GRAPH_VIEW);
    assertConvexOrientation(`vertex:${preset}`, fromVertex.poly);

    const fromFace = derivePolyFromFaceGraph(fromVertex.faceGraph, GRAPH_VIEW);
    assertConvexOrientation(`face:${preset}`, fromFace.poly);
  }

  // Regression guard: intentionally reverse one face and ensure the checker catches it.
  const broken = {
    vertices: prism.vertices,
    faces: [
      ...PRISM_FACES.slice(0, 2),
      [...PRISM_FACES[2]].reverse(),
      ...PRISM_FACES.slice(3),
    ],
  };
  const brokenReport = analyzeOrientation(broken);
  if (brokenReport.badEdgeSigns === 0 && brokenReport.negFaceDots === 0) {
    fail("broken-prism regression check did not trigger");
  }

  console.log("geometrySignInvariants: OK");
}

run();
