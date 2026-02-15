import type { NodeId } from "../../graph/types";

type Props = {
  selectedNodes: NodeId[];
  selectedEdgeId: string | null;
  pinnedCount: number;
  edgeCreateMode: boolean;
  onAddNode: () => void;
  onToggleEdgeMode: () => void;
  onAddEdge: () => void;
  onDelete: () => void;
  onPinSelected: (pin: boolean) => void;
};

export function EditPanel({
  selectedNodes,
  selectedEdgeId,
  pinnedCount,
  edgeCreateMode,
  onAddNode,
  onToggleEdgeMode,
  onAddEdge,
  onDelete,
  onPinSelected,
}: Props) {
  return (
    <div style={{ border: "1px solid #e6e6e6", borderRadius: 12, padding: 10, background: "#fff" }}>
      <div style={{ fontWeight: 700, marginBottom: 6 }}>Edit</div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button onClick={onAddNode} style={{ padding: "6px 10px", borderRadius: 10, border: "1px solid #ddd", background: "white" }}>
          Add node
        </button>
        <button
          onClick={onToggleEdgeMode}
          style={{ padding: "6px 10px", borderRadius: 10, border: "1px solid #ddd", background: edgeCreateMode ? "#f3f3f3" : "white" }}
          title="Edge creation mode: click a start node, then an end node to create an edge."
        >
          Edge mode {edgeCreateMode ? "ON" : "OFF"}
        </button>
        <button
          onClick={onAddEdge}
          disabled={selectedNodes.length !== 2}
          style={{ padding: "6px 10px", borderRadius: 10, border: "1px solid #ddd", background: "white", opacity: selectedNodes.length !== 2 ? 0.5 : 1 }}
        >
          Add edge
        </button>
        <button
          onClick={onDelete}
          disabled={selectedNodes.length === 0 && !selectedEdgeId}
          style={{ padding: "6px 10px", borderRadius: 10, border: "1px solid #ddd", background: "white", opacity: selectedNodes.length === 0 && !selectedEdgeId ? 0.5 : 1 }}
        >
          Delete
        </button>
      </div>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10, alignItems: "center" }}>
        <span style={{ fontSize: 12, color: "#444" }}>Pins ({pinnedCount})</span>
        <button
          onClick={() => onPinSelected(true)}
          disabled={selectedNodes.length === 0}
          style={{ padding: "6px 10px", borderRadius: 10, border: "1px solid #ddd", background: "white", opacity: selectedNodes.length === 0 ? 0.5 : 1 }}
        >
          Pin selected
        </button>
        <button
          onClick={() => onPinSelected(false)}
          disabled={selectedNodes.length === 0}
          style={{ padding: "6px 10px", borderRadius: 10, border: "1px solid #ddd", background: "white", opacity: selectedNodes.length === 0 ? 0.5 : 1 }}
        >
          Unpin selected
        </button>
      </div>
    </div>
  );
}
