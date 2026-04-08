from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ArtifactStreamResult:
    ela_score: float
    edge_artifact_score: float
    ai_artifact_probability: float
    ela_detail: str
    edge_detail: str
    anomaly_map: np.ndarray


def _compute_ela_map(bgr: np.ndarray, jpeg_quality: int = 90) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        return np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.float32)
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(bgr, recompressed)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return cv2.normalize(diff_gray.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)


def run_artifact_stream(bgr: np.ndarray, gray: np.ndarray) -> ArtifactStreamResult:
    ela_map = _compute_ela_map(bgr)
    ela_score = float(np.mean(ela_map))

    edges = cv2.Canny(gray, 80, 170).astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    lap_norm = cv2.normalize(np.abs(lap), None, 0.0, 1.0, cv2.NORM_MINMAX)

    edge_regions = edges > 0.1
    if np.any(edge_regions):
        edge_artifact_score = float(np.mean(lap_norm[edge_regions]))
    else:
        edge_artifact_score = float(np.mean(lap_norm))

    ai_artifact_probability = float(np.clip(0.55 * ela_score + 0.45 * edge_artifact_score, 0.0, 1.0))
    ela_detail = f"ELA mean={ela_score:.3f}; higher values indicate inconsistent recompression patterns."
    edge_detail = (
        f"Edge artifact score={edge_artifact_score:.3f}; irregular high-frequency boundaries raise synthetic risk."
    )
    anomaly_map = np.clip(0.6 * ela_map + 0.4 * lap_norm, 0.0, 1.0)
    return ArtifactStreamResult(
        ela_score=ela_score,
        edge_artifact_score=edge_artifact_score,
        ai_artifact_probability=ai_artifact_probability,
        ela_detail=ela_detail,
        edge_detail=edge_detail,
        anomaly_map=anomaly_map,
    )
