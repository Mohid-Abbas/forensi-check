import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find a best authenticity threshold from benchmark CSV.")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark CSV file")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def score_for_threshold(rows: list[dict[str, str]], threshold: float) -> float:
    correct = 0
    for row in rows:
        authenticity = float(row["authenticity_score"])
        predicted = "real" if authenticity >= threshold else "ai"
        if predicted == row["label"]:
            correct += 1
    return correct / max(len(rows), 1)


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.benchmark))
    if not rows:
        raise SystemExit("Empty benchmark file.")

    best_threshold = 50.0
    best_accuracy = 0.0
    for threshold in range(20, 81):
        acc = score_for_threshold(rows, float(threshold))
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(threshold)

    print(f"Best threshold: {best_threshold:.1f}")
    print(f"Accuracy: {best_accuracy * 100.0:.2f}%")
    print("Set FORENSICHECK_AUTHENTIC_THRESHOLD to this value for deployment calibration.")


if __name__ == "__main__":
    main()
