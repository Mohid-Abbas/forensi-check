import type { AnalyzeResponse } from "../api/client";
import type { CSSProperties } from "react";

type Props = {
  result: AnalyzeResponse;
  decisionThreshold: number;
};

export function ResultPanel({ result, decisionThreshold }: Props) {
  const thresholdBasedVerdict =
    result.authenticity_score >= decisionThreshold ? "Authentic" : "AI-Generated";
  const scoreBar = `${result.authenticity_score}%`;
  const aiBar = `${result.ai_probability * 100}%`;

  return (
    <section style={styles.box}>
      <div style={styles.headerRow}>
        <h2 style={{ margin: 0 }}>Verdict: {thresholdBasedVerdict}</h2>
        <span style={{ ...styles.badge, ...(result.decision_band === "low" ? styles.badgeWarn : styles.badgeOk) }}>
          Confidence: {result.decision_band}
        </span>
      </div>
      <p style={styles.meta}>
        API verdict: {result.verdict} | Threshold preview: {decisionThreshold.toFixed(0)}%
      </p>
      <div style={styles.metricCard}>
        <p style={styles.metricLabel}>Authenticity score: {result.authenticity_score.toFixed(2)}%</p>
        <div style={styles.track}>
          <div style={{ ...styles.fill, width: scoreBar }} />
        </div>
      </div>
      <div style={styles.metricCard}>
        <p style={styles.metricLabel}>AI probability: {(result.ai_probability * 100).toFixed(2)}%</p>
        <div style={styles.track}>
          <div style={{ ...styles.fillDanger, width: aiBar }} />
        </div>
      </div>
      <p style={styles.meta}>Latency: {result.latency_ms.toFixed(2)} ms</p>
      {!result.model_calibrated && (
        <p style={styles.warning}>
          Neural model is not calibrated yet. Add `FORENSICHECK_VIT_WEIGHTS` for reliable AI predictions.
        </p>
      )}
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
    border: "1px solid #d1d5db",
    borderRadius: 14,
    padding: 16,
    marginBottom: 16,
    backgroundColor: "#ffffff"
  },
  headerRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
    marginBottom: 6
  },
  badge: {
    fontSize: 12,
    padding: "6px 10px",
    borderRadius: 999
  } as CSSProperties,
  badgeWarn: {
    backgroundColor: "#fff7ed",
    color: "#9a3412"
  },
  badgeOk: {
    backgroundColor: "#ecfeff",
    color: "#155e75"
  },
  meta: {
    color: "#4b5563",
    margin: "6px 0 10px"
  },
  metricCard: {
    marginBottom: 10
  },
  metricLabel: {
    margin: "0 0 4px",
    fontWeight: 600
  },
  track: {
    width: "100%",
    height: 10,
    borderRadius: 999,
    backgroundColor: "#e5e7eb",
    overflow: "hidden"
  },
  fill: {
    height: "100%",
    background: "linear-gradient(90deg, #22c55e, #0ea5e9)"
  },
  fillDanger: {
    height: "100%",
    background: "linear-gradient(90deg, #f59e0b, #ef4444)"
  },
  warning: {
    backgroundColor: "#fff7ed",
    border: "1px solid #fed7aa",
    color: "#9a3412",
    borderRadius: 8,
    padding: 10
  }
};
