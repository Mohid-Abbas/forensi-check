# ForensiCheck

ForensiCheck is a dual-stream digital image forensics web app for authenticity verification.
It combines:

- **Stream A (Statistical):** sensor-noise residual extraction + entropy/frequency signals.
- **Stream B (Neural):** ViT-based AI-generation probability.

The API returns an authenticity score, verdict, forensic report, and heatmap overlay for explainable analysis.

## Repository Structure

- `backend/app/main.py`: FastAPI API entrypoint (`/health`, `/analyze`).
- `backend/app/services/noise_stream.py`: high-pass residual and entropy analysis.
- `backend/app/services/vit_stream.py`: ViT inference and confidence map generation.
- `backend/app/services/fusion.py`: weighted score fusion and verdict.
- `backend/app/services/heatmap.py`: forensic heatmap overlay generation.
- `backend/scripts/benchmark_dataset.py`: KPI benchmark script.
- `frontend/src`: React + TypeScript dashboard.

## Local Development

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Optional model weights:

- Set `FORENSICHECK_VIT_WEIGHTS` to a fine-tuned checkpoint path.
- If unset, the app uses a pre-trained backbone with a two-class head for MVP bootstrapping.

## Train Your Model (Recommended)

Place labeled images in:

- `backend/dataset/real`
- `backend/dataset/ai`

Then train:

```bash
cd backend
python scripts/train_vit.py --data-dir dataset --output models/forensic_vit_best.pth --epochs 6 --batch-size 16
```

Evaluate:

```bash
python scripts/evaluate_vit.py --data-dir dataset --weights models/forensic_vit_best.pth
```

Use in API:

```bash
FORENSICHECK_VIT_WEIGHTS=backend/models/forensic_vit_best.pth
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Set API base URL if needed:

```bash
VITE_API_BASE=http://localhost:8000
```

## API Contract

`POST /analyze` (multipart image upload) returns:

- `authenticity_score` (0-100)
- `ai_probability` (0-1)
- `verdict` (`Authentic` or `AI-Generated`)
- `forensic_report`
- `noise_signal`, `ela_signal`, `edge_signal`, `cnn_signal` (explainability signals)
- `heatmap` (base64 PNG overlay)
- `latency_ms`

If no tuned model checkpoint is configured, verdict is returned as `Inconclusive` to avoid misleading confidence.

## KPI Evaluation

1. Place labeled samples in `backend/dataset/real` and `backend/dataset/ai`.
2. Run:

```bash
python backend/scripts/benchmark_dataset.py --api-url http://localhost:8000 --dataset-dir backend/dataset --output backend/reports/benchmark.csv
```

Tracks:

- Detection accuracy target: **>90%**
- Latency target: **<3 seconds** per image
- Explainability target: at least two technical signals in every response

Threshold calibration:

```bash
python backend/scripts/calibrate_thresholds.py --benchmark backend/reports/benchmark.csv
```

Performance tuning env vars:

- `FORENSICHECK_NOISE_WEIGHT` (default `0.35`)
- `FORENSICHECK_VIT_WEIGHT` (default `0.65`)
- `FORENSICHECK_AUTHENTIC_THRESHOLD` (default `50`)

## Deployment

### Docker Compose

```bash
docker compose up --build
```

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`

### Suggested Cloud Split

- Frontend: Vercel (static build).
- Backend: Render/Fly.io/other Python host with model-weight environment config.

## Social Impact (SDG 16)

ForensiCheck supports **UN SDG 16 (Peace, Justice, and Strong Institutions)** by helping journalists, moderators, and OSINT analysts detect digital forgery and reduce misinformation harm.
