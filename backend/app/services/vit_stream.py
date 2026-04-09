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
    is_calibrated: bool


class ViTClassifier:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.is_calibrated = False
        self.ai_class_index = 0
        weights_path = os.getenv("FORENSICHECK_VIT_WEIGHTS", "").strip()
        if weights_path and os.path.exists(weights_path):
            self.model.heads.head = torch.nn.Linear(self.model.heads.head.in_features, 2)
            loaded = torch.load(weights_path, map_location=self.device)
            if isinstance(loaded, dict) and "model_state_dict" in loaded:
                state = loaded["model_state_dict"]
                class_to_idx = loaded.get("class_to_idx", {})
                if isinstance(class_to_idx, dict) and "ai" in class_to_idx:
                    self.ai_class_index = int(class_to_idx["ai"])
            else:
                # Backward compatibility with old checkpoints that only stored weights.
                state = loaded
            override_idx = os.getenv("FORENSICHECK_AI_CLASS_INDEX", "").strip()
            if override_idx:
                self.ai_class_index = int(override_idx)
            self.model.load_state_dict(state, strict=False)
            self.is_calibrated = True
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
        tensor = self.tf(rgb).unsqueeze(0).to(self.device)
        if self.is_calibrated:
            with torch.inference_mode():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
            ai_probability = float(probs[self.ai_class_index])
            detail = (
                f"ViT AI probability={ai_probability:.2f} "
                f"(fine-tuned checkpoint, ai_class_index={self.ai_class_index})."
            )
        else:
            # Without a forensic checkpoint, avoid random/unreliable class decisions.
            ai_probability = 0.5
            detail = (
                "ViT checkpoint not loaded; neural stream is running in neutral mode "
                "(ai_probability=0.50). Set FORENSICHECK_VIT_WEIGHTS for calibrated predictions."
            )

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        edges = cv2.Laplacian(gray, cv2.CV_32F)
        edge_map = cv2.normalize(np.abs(edges), None, 0.0, 1.0, cv2.NORM_MINMAX)
        confidence_map = np.clip(edge_map * max(ai_probability, 0.35), 0.0, 1.0)
        return VitResult(
            ai_probability=ai_probability,
            detail=detail,
            confidence_map=confidence_map,
            is_calibrated=self.is_calibrated,
        )


@lru_cache(maxsize=1)
def get_vit_classifier() -> ViTClassifier:
    return ViTClassifier()
