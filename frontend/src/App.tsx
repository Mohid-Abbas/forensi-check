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
    <main style={{ maxWidth: 1080, margin: "0 auto", padding: "24px 16px", fontFamily: "Arial, sans-serif" }}>
      <h1>ForensiCheck</h1>
      <p>Dual-stream image authenticity analysis for AI-generated content detection.</p>
      <UploadPanel onSelect={onSelect} isLoading={isLoading} />
      {isLoading && <p>Analyzing image...</p>}
      {error && <p style={{ color: "#cc1f1f" }}>{error}</p>}
      {result && <ResultPanel result={result} />}
      {result && sourceUrl && <HeatmapViewer sourceUrl={sourceUrl} heatmapBase64={result.heatmap} />}
    </main>
  );
}
