import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class PreprocessedImage:
    rgb: np.ndarray
    bgr: np.ndarray
    gray: np.ndarray


def decode_image(file_bytes: bytes) -> PreprocessedImage:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    rgb = np.array(image, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return PreprocessedImage(rgb=rgb, bgr=bgr, gray=gray)
