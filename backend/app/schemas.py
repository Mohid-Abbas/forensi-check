from pydantic import BaseModel, Field


class Signal(BaseModel):
    name: str
    value: float
    detail: str


class AnalyzeResponse(BaseModel):
    authenticity_score: float = Field(ge=0.0, le=100.0)
    ai_probability: float = Field(ge=0.0, le=1.0)
    verdict: str
    forensic_report: str
    noise_signal: Signal
    cnn_signal: Signal
    heatmap: str
    latency_ms: float
