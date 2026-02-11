import React from "react";
import type { SimpleGraph } from "../graph/types";
import { SingleGraphEditor } from "./graphEditor/SingleGraphEditor";

type Props = {
  faceGraph: SimpleGraph;
  vertexGraph: SimpleGraph;
  updateFaceGraph: (g: SimpleGraph) => void;
  commitFaceGraph: (g: SimpleGraph) => void;
  updateVertexGraph: (g: SimpleGraph) => void;
  commitVertexGraph: (g: SimpleGraph) => void;
};

const GRAPH_VIEW = { w: 420, h: 360, padding: 28 };

export function GraphEditor({ faceGraph, vertexGraph, updateFaceGraph, commitFaceGraph, updateVertexGraph, commitVertexGraph }: Props) {
  return (
    <div style={{ display: "grid", gap: 12 }}>
      <SingleGraphEditor title="Face graph" graph={faceGraph} updateGraph={updateFaceGraph} commitGraph={commitFaceGraph} onSyncDual={(dual) => commitVertexGraph(dual)} newNodePrefix="F" canvas={GRAPH_VIEW} />
      <SingleGraphEditor title="Vertex graph" graph={vertexGraph} updateGraph={updateVertexGraph} commitGraph={commitVertexGraph} onSyncDual={(dual) => commitFaceGraph(dual)} newNodePrefix="V" canvas={GRAPH_VIEW} />
    </div>
  );
}
