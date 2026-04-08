import os
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import torch
from torchvision import models, transforms


@dataclass
class VitResult:
    ai_probability: float
    detail: str
    confidence_map: np.ndarray


class ViTClassifier:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.model.heads.head = torch.nn.Linear(self.model.heads.head.in_features, 2)
        weights_path = os.getenv("FORENSICHECK_VIT_WEIGHTS", "").strip()
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()
        self.tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def infer(self, rgb: np.ndarray) -> VitResult:
        with torch.inference_mode():
            tensor = self.tf(rgb).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        ai_probability = float(probs[1])

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        edges = cv2.Laplacian(gray, cv2.CV_32F)
        edge_map = cv2.normalize(np.abs(edges), None, 0.0, 1.0, cv2.NORM_MINMAX)
        confidence_map = np.clip(edge_map * ai_probability, 0.0, 1.0)
        detail = f"ViT AI probability={ai_probability:.2f} (class-1 softmax confidence)."
        return VitResult(ai_probability=ai_probability, detail=detail, confidence_map=confidence_map)


@lru_cache(maxsize=1)
def get_vit_classifier() -> ViTClassifier:
    return ViTClassifier()
