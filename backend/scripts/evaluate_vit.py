import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained ViT on labeled dataset.")
    parser.add_argument("--data-dir", default="dataset", help="Directory with real/ and ai/")
    parser.add_argument("--weights", required=True, help="Path to trained .pth checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    weights_path = Path(args.weights)
    if not data_dir.exists():
        raise SystemExit(f"Dataset directory missing: {data_dir}")
    if not weights_path.exists():
        raise SystemExit(f"Weights file missing: {weights_path}")

    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = datasets.ImageFolder(str(data_dir), transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, 2)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    total = 0
    correct = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        total += int(labels.size(0))
        correct += int((preds == labels).sum().item())

    acc = correct / max(total, 1)
    print(f"Samples: {total}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Classes: {ds.class_to_idx}")


if __name__ == "__main__":
    main()
