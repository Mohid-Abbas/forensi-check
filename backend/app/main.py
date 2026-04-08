import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AnalyzeResponse, Signal
from app.services.artifact_stream import run_artifact_stream
from app.services.fusion import fuse_scores
from app.services.heatmap import generate_overlay_base64
from app.services.noise_stream import run_noise_stream
from app.services.preprocess import decode_image
from app.services.report import build_report
from app.services.vit_stream import get_vit_classifier

app = FastAPI(title="ForensiCheck API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    started = time.perf_counter()
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image = decode_image(data)
    noise = run_noise_stream(image.gray)
    artifacts = run_artifact_stream(image.bgr, image.gray)
    vit = get_vit_classifier().infer(image.rgb)
    fused = fuse_scores(
        noise.ai_noise_probability,
        vit.ai_probability,
        artifacts.ai_artifact_probability,
        vit.is_calibrated,
    )
    heatmap = generate_overlay_base64(image.rgb, noise.residual, vit.confidence_map, artifacts.anomaly_map)
    forensic_report = build_report(
        fused.verdict,
        noise.detail,
        vit.detail,
        model_calibrated=vit.is_calibrated,
        decision_band=fused.decision_band,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return AnalyzeResponse(
        authenticity_score=fused.authenticity_score,
        ai_probability=round(fused.ai_probability, 4),
        verdict=fused.verdict,
        decision_band=fused.decision_band,
        model_calibrated=vit.is_calibrated,
        forensic_report=forensic_report,
        noise_signal=Signal(
            name="Noise Residual Entropy",
            value=round(noise.entropy, 4),
            detail=noise.detail,
        ),
        ela_signal=Signal(
            name="ELA Recompression Anomaly",
            value=round(artifacts.ela_score, 4),
            detail=artifacts.ela_detail,
        ),
        edge_signal=Signal(
            name="Edge Artifact Score",
            value=round(artifacts.edge_artifact_score, 4),
            detail=artifacts.edge_detail,
        ),
        cnn_signal=Signal(
            name="ViT Confidence",
            value=round(vit.ai_probability, 4),
            detail=vit.detail,
        ),
        heatmap=heatmap,
        latency_ms=round(elapsed_ms, 2),
    )
