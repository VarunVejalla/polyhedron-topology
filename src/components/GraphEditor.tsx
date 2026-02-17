import { useState } from "react";
import type { SimpleGraph } from "../graph/types";
import { GRAPH_VIEW } from "../graph/view";
import { SingleGraphEditor } from "./graphEditor/SingleGraphEditor";

type Props = {
  faceGraph: SimpleGraph;
  vertexGraph: SimpleGraph;
  updateFaceGraph: (g: SimpleGraph) => void;
  commitFaceGraph: (g: SimpleGraph) => void;
  updateVertexGraph: (g: SimpleGraph) => void;
  commitVertexGraph: (g: SimpleGraph) => void;
};

export function GraphEditor({ faceGraph, vertexGraph, updateFaceGraph, commitFaceGraph, updateVertexGraph, commitVertexGraph }: Props) {
  const [activeTab, setActiveTab] = useState<"face" | "vertex">("face");

  return (
    <div className="graphEditorRoot">
      <div className="graphTabBar">
        <button className={`graphTab ${activeTab === "face" ? "graphTabActive" : ""}`} onClick={() => setActiveTab("face")}>
          Face graph
        </button>
        <button className={`graphTab ${activeTab === "vertex" ? "graphTabActive" : ""}`} onClick={() => setActiveTab("vertex")}>
          Vertex graph
        </button>
      </div>

      {activeTab === "face" ? (
        <SingleGraphEditor
          title="Face graph"
          graph={faceGraph}
          updateGraph={updateFaceGraph}
          commitGraph={commitFaceGraph}
          onSyncDual={(dual) => commitVertexGraph(dual)}
          newNodePrefix="F"
          canvas={GRAPH_VIEW}
        />
      ) : (
        <SingleGraphEditor
          title="Vertex graph"
          graph={vertexGraph}
          updateGraph={updateVertexGraph}
          commitGraph={commitVertexGraph}
          onSyncDual={(dual) => commitFaceGraph(dual)}
          newNodePrefix="V"
          canvas={GRAPH_VIEW}
        />
      )}
    </div>
  );
}
