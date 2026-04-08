import base64
import io

import cv2
import numpy as np
from PIL import Image


def generate_overlay_base64(rgb: np.ndarray, residual: np.ndarray, confidence_map: np.ndarray) -> str:
    residual_map = cv2.normalize(residual.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    combined = np.clip(0.55 * confidence_map + 0.45 * residual_map, 0.0, 1.0)
    heat_u8 = (combined * 255).astype(np.uint8)
    heat_colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base, 0.65, heat_colored, 0.35, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(overlay_rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
