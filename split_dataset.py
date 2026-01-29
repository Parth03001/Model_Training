"""
Step 1: Split preprocessed wheel dataset into train/val/test (70/15/15).
Stratified split — each class gets the same ratio.

Usage:
    python split_dataset.py --input Wheel_Data_Processed --output Wheel_Split
"""

import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


def split_dataset(input_dir, output_dir, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Stratified split of image folders into train/val/test.
    Preserves class folder structure under each split.
    """
    random.seed(seed)
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Collect images per class (skip _preview folder)
    class_images = defaultdict(list)
    for cls_dir in sorted(input_path.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith('_'):
            continue
        for img in cls_dir.iterdir():
            if img.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                class_images[cls_dir.name].append(img)

    if not class_images:
        print(f"ERROR: No class folders found in {input_dir}")
        return

    # Create output structure
    splits = ['train', 'val', 'test']
    for split in splits:
        for cls_name in class_images:
            (output_path / split / cls_name).mkdir(parents=True, exist_ok=True)

    # Split each class
    total_counts = {'train': 0, 'val': 0, 'test': 0}

    for cls_name, images in sorted(class_images.items()):
        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for img in train_imgs:
            shutil.copy2(img, output_path / 'train' / cls_name / img.name)
        for img in val_imgs:
            shutil.copy2(img, output_path / 'val' / cls_name / img.name)
        for img in test_imgs:
            shutil.copy2(img, output_path / 'test' / cls_name / img.name)

        total_counts['train'] += len(train_imgs)
        total_counts['val'] += len(val_imgs)
        total_counts['test'] += len(test_imgs)

        print(f"  {cls_name}: {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test  (total {n})")

    print(f"\nSplit complete:")
    for split in splits:
        print(f"  {output_path / split}/  ({total_counts[split]} images)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--input", type=str, default="Wheel_Data_Processed",
                        help="Input folder with class subfolders")
    parser.add_argument("--output", type=str, default="Wheel_Split",
                        help="Output folder for split dataset")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(args.input, args.output,
                  train_ratio=args.train_ratio,
                  val_ratio=args.val_ratio,
                  seed=args.seed)
