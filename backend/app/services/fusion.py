from dataclasses import dataclass

from app.services.config import AUTHENTIC_THRESHOLD, NOISE_WEIGHT, VIT_WEIGHT


@dataclass
class FusionResult:
    ai_probability: float
    authenticity_score: float
    verdict: str


def fuse_scores(noise_ai_probability: float, vit_ai_probability: float) -> FusionResult:
    ai_probability = max(
        0.0,
        min(1.0, (NOISE_WEIGHT * noise_ai_probability) + (VIT_WEIGHT * vit_ai_probability)),
    )
    authenticity_score = (1.0 - ai_probability) * 100.0
    verdict = "Authentic" if authenticity_score >= AUTHENTIC_THRESHOLD else "AI-Generated"
    return FusionResult(
        ai_probability=ai_probability,
        authenticity_score=round(authenticity_score, 2),
        verdict=verdict,
    )
