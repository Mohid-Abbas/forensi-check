import type { AnalyzeResponse } from "../api/client";

type Props = {
  result: AnalyzeResponse;
};

export function ResultPanel({ result }: Props) {
  return (
    <section style={styles.box}>
      <h2>Verdict: {result.verdict}</h2>
      <p>Authenticity score: {result.authenticity_score.toFixed(2)}%</p>
      <p>AI probability: {(result.ai_probability * 100).toFixed(2)}%</p>
      <p>Latency: {result.latency_ms.toFixed(2)} ms</p>
      <p>{result.forensic_report}</p>
      <ul>
        <li>
          <strong>{result.noise_signal.name}:</strong> {result.noise_signal.value.toFixed(4)} -{" "}
          {result.noise_signal.detail}
        </li>
        <li>
          <strong>{result.cnn_signal.name}:</strong> {result.cnn_signal.value.toFixed(4)} -{" "}
          {result.cnn_signal.detail}
        </li>
      </ul>
    </section>
  );
}

const styles = {
  box: {
    border: "1px solid #303030",
    borderRadius: 8,
    padding: 16,
    marginBottom: 16
  }
};
