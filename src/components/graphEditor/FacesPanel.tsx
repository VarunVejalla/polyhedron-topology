import type { Face } from "../../graph/embedding";

type Props = {
  isPlanarOk: boolean;
  faces: Face[] | null;
  polyFaceError: string | null;
};

export function FacesPanel({ isPlanarOk, faces, polyFaceError }: Props) {
  return (
    <div style={{ border: "1px solid #e6e6e6", borderRadius: 12, padding: 10, background: "#fff" }}>
      <div style={{ fontWeight: 700, marginBottom: 6 }}>Faces</div>
      {!isPlanarOk && <div style={{ fontSize: 12, color: "#666" }}>Faces require a polyhedral embedding.</div>}
      {isPlanarOk && !faces && !polyFaceError && <div style={{ fontSize: 12, color: "#666" }}>No faces available yet.</div>}
      {isPlanarOk && !faces && polyFaceError && (
        <div style={{ fontSize: 12, color: "#c44" }}>
          <div>Face extraction failed:</div>
          <div style={{ marginTop: 4, fontFamily: "monospace", fontSize: 11, background: "#fdd", padding: 6, borderRadius: 6 }}>{polyFaceError}</div>
        </div>
      )}
      {isPlanarOk && faces && (
        <div style={{ display: "grid", gap: 6, fontSize: 12, maxHeight: 220, overflow: "auto" }}>
          {faces.map((f) => (
            <div key={f.id} style={{ padding: "6px 8px", borderRadius: 10, border: "1px solid #eee", background: "#fafafa" }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ fontWeight: 600 }}>{f.id}</span>
                <span style={{ color: "#666" }}>|cycle|={f.cycle.length}</span>
              </div>
              <div style={{ color: "#444", marginTop: 4, wordBreak: "break-word" }}>{f.cycle.join(" ")}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
