import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import requests


@dataclass
class BenchmarkResult:
    filename: str
    label: str
    predicted: str
    authenticity_score: float
    latency_ms: float

    @property
    def is_correct(self) -> bool:
        return self.label == self.predicted


def iter_images(dataset_dir: Path) -> Iterable[tuple[Path, str]]:
    for label in ("real", "ai"):
        class_dir = dataset_dir / label
        if not class_dir.exists():
            continue
        for item in class_dir.iterdir():
            if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                yield item, label


def run_benchmark(api_url: str, dataset_dir: Path, timeout: int) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    for image_path, label in iter_images(dataset_dir):
        with image_path.open("rb") as fh:
            started = time.perf_counter()
            response = requests.post(
                f"{api_url.rstrip('/')}/analyze",
                files={"file": (image_path.name, fh, "application/octet-stream")},
                timeout=timeout,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        payload = response.json()
        verdict = str(payload.get("verdict", "")).lower()
        predicted = "real" if verdict == "authentic" else "ai"
        results.append(
            BenchmarkResult(
                filename=image_path.name,
                label=label,
                predicted=predicted,
                authenticity_score=float(payload.get("authenticity_score", 0.0)),
                latency_ms=latency_ms,
            )
        )
    return results


def write_report(output_path: Path, results: List[BenchmarkResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            ["filename", "label", "predicted", "authenticity_score", "latency_ms", "is_correct"]
        )
        for row in results:
            writer.writerow(
                [
                    row.filename,
                    row.label,
                    row.predicted,
                    f"{row.authenticity_score:.2f}",
                    f"{row.latency_ms:.2f}",
                    int(row.is_correct),
                ]
            )


def print_summary(results: List[BenchmarkResult]) -> None:
    if not results:
        print("No images found. Place files in dataset/{real,ai}/")
        return
    accuracy = sum(x.is_correct for x in results) / len(results) * 100.0
    avg_latency = sum(x.latency_ms for x in results) / len(results)
    p95_idx = max(int(len(results) * 0.95) - 1, 0)
    sorted_lat = sorted(x.latency_ms for x in results)
    p95 = sorted_lat[p95_idx]
    print(f"Samples: {len(results)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"P95 latency: {p95:.2f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ForensiCheck API on a labeled dataset.")
    parser.add_argument("--api-url", default="http://localhost:8000", help="FastAPI server base URL")
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing real/ and ai/")
    parser.add_argument("--output", default="reports/benchmark.csv", help="CSV output path")
    parser.add_argument("--timeout", default=30, type=int, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")
    results = run_benchmark(args.api_url, dataset_dir, args.timeout)
    write_report(Path(args.output), results)
    print_summary(results)


if __name__ == "__main__":
    main()
