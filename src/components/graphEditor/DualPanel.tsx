import React from "react";

type Props = {
  canSync: boolean;
  onSyncDual: () => void;
};

export function DualPanel({ canSync, onSyncDual }: Props) {
  return (
    <div style={{ border: "1px solid #e6e6e6", borderRadius: 12, padding: 10, background: "#fff" }}>
      <div style={{ fontWeight: 700, marginBottom: 6 }}>Dual sync</div>
      <div style={{ fontSize: 12, color: "#444", lineHeight: 1.35 }}>
        When this graph is polyhedral, you can sync the other panel to its planar dual (stable by face-centroid positions).
      </div>
      <button onClick={onSyncDual} disabled={!canSync} style={{ marginTop: 10, padding: "6px 10px", borderRadius: 10, border: "1px solid #ddd", background: "white", opacity: !canSync ? 0.5 : 1 }}>
        Sync dual
      </button>
    </div>
  );
}
