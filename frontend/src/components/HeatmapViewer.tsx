import { useState } from "react";

type Props = {
  sourceUrl: string;
  heatmapBase64: string;
};

export function HeatmapViewer({ sourceUrl, heatmapBase64 }: Props) {
  const [showBlend, setShowBlend] = useState(true);

  return (
    <section style={styles.grid}>
      <div>
        <h3>Original</h3>
        <img src={sourceUrl} alt="Uploaded source" style={styles.image} />
      </div>
      <div>
        <h3>Forensic Heatmap</h3>
        <label style={styles.toggle}>
          <input type="checkbox" checked={showBlend} onChange={() => setShowBlend((v) => !v)} /> Show overlay
        </label>
        <img
          src={showBlend ? `data:image/png;base64,${heatmapBase64}` : sourceUrl}
          alt="Forensic heatmap overlay"
          style={styles.image}
        />
      </div>
    </section>
  );
}

const styles = {
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
    gap: 16
  },
  image: {
    width: "100%",
    borderRadius: 8,
    border: "1px solid #2f2f2f"
  },
  toggle: {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    marginBottom: 8
  }
};
