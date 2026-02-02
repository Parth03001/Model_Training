"""
Step 2: EfficientNetV2-S Training for 4-Class Wheel Classification

Setup:
- EfficientNetV2-S backbone (384x384 input, 21.5M params)
- Freeze backbone for first N epochs, then unfreeze with lower lr
- YOLO-style augmentation + label smoothing
- Early stopping + dropout on classifier head
- Cosine annealing lr schedule

Usage:
    python train_wheel_efficientnet.py --data Wheel_Split --epochs 50
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import argparse
from collections import Counter


# ── Dataset ──────────────────────────────────────────────────────────────────

class WheelDataset(Dataset):
    """Load images from split folder structure: split/class_name/image.jpg"""

    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_names = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir()
        ])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        for cls_name in self.class_names:
            cls_dir = self.root / cls_name
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Transforms ───────────────────────────────────────────────────────────────

IMG_SIZE = 384  # EfficientNetV2-S native input resolution


def get_transforms(is_train=False):
    """
    YOLO-style augmentation for training; plain normalize for val/test.
    Images are preprocessed at 384x384 for EfficientNetV2-S.

    Augmentations inspired by YOLO training:
    - Rotation up to 20 degrees
    - Random zoom in/out via RandomResizedCrop
    - Horizontal & vertical flip
    - Color jitter (brightness, contrast, saturation)
    - Random affine (small translate + shear)
    - Random erasing for regularization
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_train:
        return transforms.Compose([
            # Zoom in/out: crop between 80-100% of image then resize back
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.80, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.RandomAffine(
                degrees=0, translate=(0.08, 0.08), shear=5
            ),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02
            ),
            transforms.ToTensor(),
            normalize,
            # Random erasing like YOLO cutout
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            normalize,
        ])


# ── Model ────────────────────────────────────────────────────────────────────

def build_model(num_classes=4, dropout=0.3):
    """EfficientNetV2-S with custom classifier head."""
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model


def freeze_backbone(model):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model):
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, 100.0 * correct / total, all_preds, all_labels


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_dir = Path(args.data) / "train"
    val_dir = Path(args.data) / "val"
    test_dir = Path(args.data) / "test"

    train_ds = WheelDataset(train_dir, transform=get_transforms(is_train=True))
    test_ds = WheelDataset(test_dir, transform=get_transforms(is_train=False))

    # Val may be empty for tiny datasets — fall back to test for monitoring
    has_val = val_dir.exists() and any(val_dir.iterdir())
    if has_val:
        val_ds = WheelDataset(val_dir, transform=get_transforms(is_train=False))
    else:
        print("WARNING: Val set empty, using test set for validation monitoring.")
        val_ds = test_ds

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val:   {len(val_ds)}")
    print(f"  Test:  {len(test_ds)}")
    print(f"  Classes: {train_ds.class_names}")

    train_labels = [s[1] for s in train_ds.samples]
    dist = Counter(train_labels)
    print(f"  Train distribution: { {train_ds.class_names[k]: v for k, v in sorted(dist.items())} }")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    num_classes = len(train_ds.class_names)
    model = build_model(num_classes=num_classes, dropout=args.dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Phase 1: Frozen backbone ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1: Frozen backbone (epochs 1-{args.freeze_epochs})")
    print(f"  lr={args.lr}, batch_size={args.batch_size}")
    print(f"{'='*60}")

    freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    patience_counter = 0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_history = []

    for epoch in range(1, args.freeze_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"[{epoch}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}%")

        all_history.append({
            "epoch": epoch, "phase": "frozen",
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

    # ── Phase 2: Unfrozen backbone (fine-tune) ───────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2: Full fine-tune (epochs {args.freeze_epochs+1}-{args.epochs})")
    print(f"  lr={args.lr_finetune}, batch_size={args.batch_size}")
    print(f"{'='*60}")

    unfreeze_backbone(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.freeze_epochs
    )

    patience_counter = 0

    for epoch in range(args.freeze_epochs + 1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{epoch}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}% | lr={current_lr:.6f}")

        all_history.append({
            "epoch": epoch, "phase": "finetune",
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # ── Test Evaluation ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Final Test Evaluation (best model)")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(save_dir / "best_model.pth", weights_only=True))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=train_ds.class_names, zero_division=0))

    print("Confusion Matrix:")
    cm = confusion_matrix(labels, preds)
    header = "            " + "  ".join(f"{n[:10]:>10}" for n in train_ds.class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>10}" for v in row)
        print(f"{train_ds.class_names[i][:10]:>10}  {row_str}")

    meta = {
        "class_names": train_ds.class_names,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "history": all_history,
        "args": vars(args)
    }
    with open(save_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved to: {save_dir / 'best_model.pth'}")
    print(f"Metadata saved to: {save_dir / 'training_meta.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNetV2-S for wheel classification")
    parser.add_argument("--data", type=str, default="Wheel_Split", help="Split dataset folder")
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs (default 50)")
    parser.add_argument("--freeze-epochs", type=int, default=8, help="Epochs with frozen backbone (default 8)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default 8)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for frozen phase (default 5e-5)")
    parser.add_argument("--lr-finetune", type=float, default=5e-6, help="Learning rate for fine-tune phase (default 5e-6)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (default 0.01)")
    parser.add_argument("--dropout", type=float, default=0.35, help="Classifier dropout (default 0.35)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default 10)")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers (default 2)")
    parser.add_argument("--save-dir", type=str, default="wheel_model_output", help="Save directory")
    args = parser.parse_args()

    main(args)
