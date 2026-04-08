# Dataset Protocol

Create this structure for benchmark runs:

- `dataset/real/`: real-camera images (news photos, phone cameras, DSLR).
- `dataset/ai/`: generated images (Diffusion/GAN outputs).

Guidelines:

1. Keep classes balanced (same number of `real` and `ai` samples).
2. Include multiple generators in `ai` (Midjourney, DALL-E, Stable Diffusion).
3. Include multiple camera/device sources in `real`.
4. Use images between `512x512` and `2048x2048` for realistic latency tests.
5. Do not include duplicates across train/validation/test splits.

Run benchmark:

`python backend/scripts/benchmark_dataset.py --api-url http://localhost:8000 --dataset-dir backend/dataset --output backend/reports/benchmark.csv`
