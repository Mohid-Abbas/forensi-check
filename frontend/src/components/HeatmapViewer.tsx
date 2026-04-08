type Props = {
  sourceUrl: string;
  heatmapBase64: string;
};

export function HeatmapViewer({ sourceUrl, heatmapBase64 }: Props) {
  return (
    <section style={styles.grid}>
      <div>
        <h3>Original</h3>
        <img src={sourceUrl} alt="Uploaded source" style={styles.image} />
      </div>
      <div>
        <h3>Forensic Heatmap</h3>
        <img src={`data:image/png;base64,${heatmapBase64}`} alt="Forensic heatmap overlay" style={styles.image} />
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
  }
};
