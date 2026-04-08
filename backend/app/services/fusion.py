from dataclasses import dataclass

from app.services.config import AUTHENTIC_THRESHOLD, NOISE_WEIGHT, VIT_WEIGHT


@dataclass
class FusionResult:
    ai_probability: float
    authenticity_score: float
    verdict: str
    decision_band: str


def fuse_scores(
    noise_ai_probability: float,
    vit_ai_probability: float,
    artifact_ai_probability: float,
    vit_calibrated: bool,
) -> FusionResult:
    noise_weight = NOISE_WEIGHT
    vit_weight = VIT_WEIGHT if vit_calibrated else 0.20
    artifact_weight = 0.25
    total_weight = noise_weight + vit_weight + artifact_weight
    ai_probability = max(
        0.0,
        min(
            1.0,
            (
                (noise_weight * noise_ai_probability)
                + (vit_weight * vit_ai_probability)
                + (artifact_weight * artifact_ai_probability)
            )
            / max(total_weight, 1e-6),
        ),
    )
    authenticity_score = (1.0 - ai_probability) * 100.0
    verdict = "Inconclusive" if not vit_calibrated else (
        "Authentic" if authenticity_score >= AUTHENTIC_THRESHOLD else "AI-Generated"
    )
    margin = abs(authenticity_score - AUTHENTIC_THRESHOLD)
    decision_band = "high" if margin >= 15 else "medium" if margin >= 7 else "low"
    return FusionResult(
        ai_probability=ai_probability,
        authenticity_score=round(authenticity_score, 2),
        verdict=verdict,
        decision_band=decision_band,
    )
