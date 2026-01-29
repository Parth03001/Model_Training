"""
Step 1: Preprocessing Pipeline for Wheel Classification
- Center-crop the hub area (rim + cap region)
- Apply light CLAHE to normalize factory lighting
- Save processed images for visual confirmation before training

Usage:
    python preprocess_wheels.py --input Wheel_Data --output Wheel_Data_Processed
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def center_crop_hub(image, crop_ratio=0.55):
    """
    Crop the center hub region from a wheel image.
    The hub (rim spokes + center cap) is roughly in the center.

    Args:
        image: BGR image
        crop_ratio: fraction of image to keep (0.55 = center 55%)
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    margin_x = int(w * (1 - crop_ratio) / 2)
    margin_y = int(h * (1 - crop_ratio) / 2)
    cropped = image[margin_y:h - margin_y, margin_x:w - margin_x]
    return cropped


def apply_light_clahe(image, clip_limit=2.0, tile_size=8):
    """
    Apply light CLAHE on L channel only (LAB space).
    Normalizes lighting without distorting colors.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def process_dataset(input_dir, output_dir, crop_ratio=0.55, clahe_clip=2.0, target_size=224):
    """
    Process all images: crop hub -> light CLAHE -> resize.
    Maintains folder structure.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Also save a side-by-side preview for visual confirmation
    preview_dir = output_path / "_preview"
    preview_dir.mkdir(exist_ok=True)

    categories = [d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith("_")]

    total = 0
    for cat_dir in sorted(categories):
        cat_output = output_path / cat_dir.name
        cat_output.mkdir(exist_ok=True)

        images = [f for f in cat_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        print(f"\n[{cat_dir.name}] Found {len(images)} images")

        for img_file in sorted(images):
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"  SKIP (unreadable): {img_file.name}")
                continue

            # Step 1: Center crop
            cropped = center_crop_hub(img, crop_ratio)

            # Step 2: Light CLAHE
            enhanced = apply_light_clahe(cropped, clip_limit=clahe_clip)

            # Step 3: Resize for model input
            resized = cv2.resize(enhanced, (target_size, target_size), interpolation=cv2.INTER_AREA)

            # Save processed image
            out_file = cat_output / img_file.name
            cv2.imwrite(str(out_file), resized)

            # Save side-by-side preview: original | cropped | cropped+CLAHE | final
            orig_small = cv2.resize(img, (target_size, target_size))
            crop_small = cv2.resize(cropped, (target_size, target_size))
            preview = np.hstack([orig_small, crop_small, resized])

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(preview, "Original", (10, 20), font, 0.5, (0, 255, 0), 1)
            cv2.putText(preview, "Cropped", (target_size + 10, 20), font, 0.5, (0, 255, 0), 1)
            cv2.putText(preview, "CLAHE", (target_size * 2 + 10, 20), font, 0.5, (0, 255, 0), 1)

            preview_file = preview_dir / f"{cat_dir.name}_{img_file.name}"
            cv2.imwrite(str(preview_file), preview)

            total += 1
            print(f"  OK: {img_file.name}")

    print(f"\nDone. Processed {total} images.")
    print(f"  Processed images: {output_path}")
    print(f"  Preview comparisons: {preview_dir}")
    print(f"\n>>> Review the _preview folder to confirm preprocessing looks good <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess wheel images: crop hub + light CLAHE")
    parser.add_argument("--input", type=str, default="Wheel_Data", help="Input dataset folder")
    parser.add_argument("--output", type=str, default="Wheel_Data_Processed", help="Output folder")
    parser.add_argument("--crop-ratio", type=float, default=0.55, help="Center crop ratio (default 0.55)")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit (default 2.0, light)")
    parser.add_argument("--size", type=int, default=224, help="Output image size (default 224)")
    args = parser.parse_args()

    process_dataset(args.input, args.output, args.crop_ratio, args.clahe_clip, args.size)
