import argparse
import copy
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ForensiCheck ViT on real vs ai images.")
    parser.add_argument("--data-dir", default="dataset", help="Directory with class folders: real/, ai/")
    parser.add_argument("--output", default="models/forensic_vit_best.pth", help="Path to save best checkpoint")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_indices(total: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(total * val_split))
    return indices[val_size:], indices[:val_size]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == labels).sum().item())
        total_items += int(labels.size(0))

    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for images, labels in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == labels).sum().item())
        total_items += int(labels.size(0))

    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    base_ds = datasets.ImageFolder(str(data_dir))
    if len(base_ds.classes) != 2 or "real" not in base_ds.class_to_idx or "ai" not in base_ds.class_to_idx:
        raise SystemExit("Dataset must contain two folders exactly: real/ and ai/")

    train_idx, val_idx = split_indices(len(base_ds), args.val_split, args.seed)
    train_ds = datasets.ImageFolder(str(data_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(data_dir), transform=val_tf)
    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    print(f"Classes: {base_ds.class_to_idx}")
    print(f"Train samples: {len(train_subset)} | Val samples: {len(val_subset)} | Device: {device}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_path)
    print(f"Saved best checkpoint to: {output_path}")
    print(f"Best validation accuracy: {best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
