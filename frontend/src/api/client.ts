export type Signal = {
  name: string;
  value: number;
  detail: string;
};

export type AnalyzeResponse = {
  authenticity_score: number;
  ai_probability: number;
  verdict: string;
  decision_band: "low" | "medium" | "high";
  model_calibrated: boolean;
  forensic_report: string;
  noise_signal: Signal;
  cnn_signal: Signal;
  heatmap: string;
  latency_ms: number;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function analyzeImage(file: File): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    throw new Error(`Analysis failed (${response.status})`);
  }
  return (await response.json()) as AnalyzeResponse;
}
