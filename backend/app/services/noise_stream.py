from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class NoiseStreamResult:
    residual: np.ndarray
    entropy: float
    high_freq_ratio: float
    ai_noise_probability: float
    detail: str


def _shannon_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    prob = hist / np.maximum(hist.sum(), 1.0)
    nz = prob[prob > 0]
    return float(-(nz * np.log2(nz)).sum())


def run_noise_stream(gray: np.ndarray) -> NoiseStreamResult:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
    residual = cv2.subtract(gray, blurred)
    residual_u8 = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    entropy = _shannon_entropy(residual_u8)

    fft = np.fft.fftshift(np.fft.fft2(residual.astype(np.float32)))
    mag = np.log1p(np.abs(fft))
    h, w = mag.shape
    center_h, center_w = h // 2, w // 2
    inner = mag[center_h - h // 8 : center_h + h // 8, center_w - w // 8 : center_w + w // 8]
    high_energy = float(np.sum(mag) - np.sum(inner))
    total_energy = float(np.sum(mag) + 1e-6)
    high_freq_ratio = high_energy / total_energy

    # Higher entropy and flatter high-frequency distribution often indicate synthetic generation.
    ai_noise_probability = float(np.clip((entropy - 4.0) / 4.0 * 0.6 + high_freq_ratio * 0.8, 0.0, 1.0))
    detail = (
        f"Residual entropy={entropy:.2f}, high-frequency ratio={high_freq_ratio:.2f}; "
        "uniform residual patterns increase synthetic likelihood."
    )
    return NoiseStreamResult(
        residual=residual_u8,
        entropy=entropy,
        high_freq_ratio=high_freq_ratio,
        ai_noise_probability=ai_noise_probability,
        detail=detail,
    )
