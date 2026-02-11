import React from "react";

type LayoutMode = "manual" | "spring" | "tutte";

type Props = {
  isPlanarOk: boolean;
  layoutMode: LayoutMode;
  setLayoutMode: (m: LayoutMode) => void;
  onApplyLayout: () => void;
  polyFaceError: string | null;
};

export function LayoutPanel({ isPlanarOk, layoutMode, setLayoutMode, onApplyLayout, polyFaceError }: Props) {
  return (
    <div style={{ border: "1px solid #e6e6e6", borderRadius: 12, padding: 10, background: "#fff" }}>
      <div style={{ fontWeight: 700, marginBottom: 6 }}>Embedding vs layout</div>
      <div style={{ fontSize: 12, color: "#444", lineHeight: 1.35 }}>Planarity is computed combinatorially (rotation system). Layout is a separate choice.</div>

      <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center", flexWrap: "wrap" }}>
        <select value={layoutMode} onChange={(e) => setLayoutMode(e.target.value as LayoutMode)} style={{ padding: "6px 8px", borderRadius: 10, border: "1px solid #ddd" }}>
          <option value="manual">manual</option>
          <option value="spring">spring</option>
          <option value="tutte" disabled={!isPlanarOk}>
            Tutte (requires planar)
          </option>
        </select>
        <button
          onClick={onApplyLayout}
          disabled={layoutMode === "manual" || (layoutMode === "tutte" && !isPlanarOk)}
          style={{ padding: "6px 10px", borderRadius: 10, border: "1px solid #ddd", background: "white", opacity: layoutMode === "manual" || (layoutMode === "tutte" && !isPlanarOk) ? 0.5 : 1 }}
        >
          Apply layout
        </button>
      </div>

      {layoutMode === "tutte" && isPlanarOk && polyFaceError && (
        <div style={{ marginTop: 10, fontSize: 12, color: "#c44" }}>
          <div>Face extraction from planarity embedding failed:</div>
          <div style={{ marginTop: 4, fontFamily: "monospace", fontSize: 11, background: "#fdd", padding: 6, borderRadius: 6 }}>{polyFaceError}</div>
        </div>
      )}

    </div>
  );
}
