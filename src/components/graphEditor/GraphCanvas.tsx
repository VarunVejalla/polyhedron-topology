import React, { useRef } from "react";
import type { GraphNode, NodeId, SimpleGraph } from "../../graph/types";

type Props = {
  graph: SimpleGraph;
  width: number;
  height: number;
  selectedNodes: NodeId[];
  selectedEdgeId: string | null;
  pinned: Set<NodeId>;
  edgeCreateMode: boolean;
  pendingEdgeStart: NodeId | null;
  onBackgroundDown: () => void;
  onPointerMove: (p: { x: number; y: number }) => void;
  onPointerUp: () => void;
  onNodeDown: (id: NodeId, p: { x: number; y: number }, meta: { shiftKey: boolean; pointerId: number }) => void;
  onEdgeClick: (eid: string) => void;
};

export function GraphCanvas({
  graph,
  width,
  height,
  selectedNodes,
  selectedEdgeId,
  pinned,
  edgeCreateMode,
  pendingEdgeStart,
  onBackgroundDown,
  onPointerMove,
  onPointerUp,
  onNodeDown,
  onEdgeClick,
}: Props) {
  const nodeById = new Map<NodeId, GraphNode>(graph.nodes.map((n) => [n.id, n]));
  const svgRef = useRef<SVGSVGElement | null>(null);

  const getLocalPoint = (e: React.PointerEvent): { x: number; y: number } => {
    const svg = svgRef.current;
    if (!svg) return { x: e.clientX, y: e.clientY };
    const r = svg.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      style={{ border: "1px solid #ddd", borderRadius: 12, background: "white", touchAction: "none" }}
      onPointerDown={() => {
        // Background click clears selection.
        // If the user is mid-drag, pointer capture will keep move events flowing.
        onBackgroundDown();
      }}
      onPointerMove={(e) => {
        onPointerMove(getLocalPoint(e));
      }}
      onPointerUp={() => {
        onPointerUp();
      }}
    >
      {graph.edges.map((e) => {
        const a = nodeById.get(e.source);
        const b = nodeById.get(e.target);
        if (!a || !b) return null;
        const selected = selectedEdgeId === e.id;
        return (
          <line
            key={e.id}
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            stroke={selected ? "#ff6600" : "#999"}
            strokeWidth={selected ? 3 : 2}
            onPointerDown={(evt) => {
              evt.stopPropagation();
              onEdgeClick(e.id);
            }}
          />
        );
      })}

      {graph.nodes.map((n) => {
        const isSel = selectedNodes.includes(n.id);
        const isPinned = pinned.has(n.id);
        const isPending = edgeCreateMode && pendingEdgeStart === n.id;
        return (
          <g
            key={n.id}
            transform={`translate(${n.x},${n.y})`}
            onPointerDown={(e) => {
              e.stopPropagation();
              // Pointer capture makes drags robust even if the cursor leaves the SVG.
              try {
                (svgRef.current as SVGSVGElement)?.setPointerCapture?.(e.pointerId);
              } catch {
                // ignore
              }
              onNodeDown(n.id, getLocalPoint(e), { shiftKey: e.shiftKey, pointerId: e.pointerId });
            }}
          >
            <circle r={12} fill={isSel ? "#ffe6cc" : "#e8f0ff"} stroke={isPending ? "#ff6600" : isPinned ? "#0a7" : "#666"} strokeWidth={isPending ? 4 : isPinned ? 3 : 2} />
            <text textAnchor="middle" dominantBaseline="central" fontSize={10} fill="#111">
              {n.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
