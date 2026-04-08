import { useState } from "react";
import { analyzeImage, type AnalyzeResponse } from "./api/client";
import { HeatmapViewer } from "./components/HeatmapViewer";
import { ResultPanel } from "./components/ResultPanel";
import { UploadPanel } from "./components/UploadPanel";

export default function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [thresholdPreview, setThresholdPreview] = useState(50);

  const onSelect = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setSourceUrl(URL.createObjectURL(file));
    try {
      const next = await analyzeImage(file);
      setResult(next);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main
      style={{
        maxWidth: 1080,
        margin: "0 auto",
        padding: "24px 16px",
        fontFamily: "Inter, Arial, sans-serif",
        color: "#111827"
      }}
    >
      <h1 style={{ marginBottom: 8 }}>ForensiCheck</h1>
      <p style={{ marginTop: 0, color: "#4b5563" }}>
        Dual-stream image authenticity analysis for AI-generated content detection.
      </p>
      <UploadPanel onSelect={onSelect} isLoading={isLoading} />
      <section
        style={{
          backgroundColor: "#f8fafc",
          border: "1px solid #e2e8f0",
          borderRadius: 12,
          padding: 12,
          marginBottom: 16
        }}
      >
        <label htmlFor="threshold" style={{ fontWeight: 600 }}>
          Decision Threshold Preview: {thresholdPreview}%
        </label>
        <input
          id="threshold"
          type="range"
          min={35}
          max={70}
          value={thresholdPreview}
          onChange={(event) => setThresholdPreview(Number(event.target.value))}
          style={{ width: "100%" }}
        />
      </section>
      {isLoading && <p>Analyzing image...</p>}
      {error && <p style={{ color: "#cc1f1f" }}>{error}</p>}
      {result && <ResultPanel result={result} decisionThreshold={thresholdPreview} />}
      {result && sourceUrl && <HeatmapViewer sourceUrl={sourceUrl} heatmapBase64={result.heatmap} />}
    </main>
  );
}
