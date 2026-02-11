export type NodeId = string;

export type GraphNode = {
  id: NodeId;
  label: string;
  x: number;
  y: number;
};

export type GraphEdge = {
  id: string;
  source: NodeId;
  target: NodeId;
};

export type SimpleGraph = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};

// A combinatorial embedding (rotation system): cyclic neighbor order around each vertex.
export type RotationSystem = Record<NodeId, NodeId[]>;

export type PolyhedralCheck =
  | { ok: true; embedding: RotationSystem }
  | { ok: false; reason: "too_small" | "nonplanar" | "not_3_connected" };
