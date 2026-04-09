import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ForensiCheck checkpoint metadata.")
    parser.add_argument("--weights", required=True, help="Path to checkpoint (.pth)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.weights)
    if not path.exists():
        raise SystemExit(f"Weights file not found: {path}")

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        class_to_idx = obj.get("class_to_idx", {})
        meta = obj.get("meta", {})
        print("checkpoint_format: metadata_bundle")
        print(f"class_to_idx: {class_to_idx}")
        print(f"meta: {meta}")
        if isinstance(class_to_idx, dict) and "ai" in class_to_idx:
            print(f"suggested FORENSICHECK_AI_CLASS_INDEX: {class_to_idx['ai']}")
    else:
        print("checkpoint_format: raw_state_dict")
        print("class_to_idx: unavailable")
        print("Set FORENSICHECK_AI_CLASS_INDEX manually (usually 0 for ImageFolder ai/real).")


if __name__ == "__main__":
    main()
