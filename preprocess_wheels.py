"""
Step 1: Preprocessing Pipeline for Wheel Classification
- Center-crop the hub area (rim + cap region)
- Apply CLAHE to normalize factory lighting
- Apply gamma correction to separate grey vs black tones
- Normalize color channels to reduce lighting bias
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


def apply_clahe(image, clip_limit=3.0, tile_size=8):
    """
    Apply CLAHE on L channel (LAB space) to normalize lighting.
    Slightly stronger clip to better separate grey vs black tones.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(image, gamma=0.85):
    """
    Apply gamma correction to brighten mid-tones.
    Gamma < 1.0 lifts grey tones away from black, making the
    difference between grey cap and black cap more visible.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")
    return cv2.LUT(image, table)


def normalize_white_balance(image):
    """
    Simple grey-world white balance to reduce color cast from factory lighting.
    Helps keep grey looking grey instead of shifting toward blue/black.
    """
    result = image.copy().astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_all / (avg_b + 1e-6)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_all / (avg_g + 1e-6)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_all / (avg_r + 1e-6)), 0, 255)
    return result.astype(np.uint8)


def process_dataset(input_dir, output_dir, crop_ratio=0.55, clahe_clip=3.0, target_size=224):
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

            # Step 2: White balance to reduce color cast
            balanced = normalize_white_balance(cropped)

            # Step 3: CLAHE to normalize lighting
            enhanced = apply_clahe(balanced, clip_limit=clahe_clip)

            # Step 4: Gamma correction to lift grey tones away from black
            corrected = apply_gamma_correction(enhanced, gamma=0.85)

            # Step 5: Resize for model input
            resized = cv2.resize(corrected, (target_size, target_size), interpolation=cv2.INTER_AREA)

            # Save processed image
            out_file = cat_output / img_file.name
            cv2.imwrite(str(out_file), resized)

            # Save side-by-side preview: original | cropped | final
            orig_small = cv2.resize(img, (target_size, target_size))
            crop_small = cv2.resize(cropped, (target_size, target_size))
            preview = np.hstack([orig_small, crop_small, resized])

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(preview, "Original", (10, 20), font, 0.5, (0, 255, 0), 1)
            cv2.putText(preview, "Cropped", (target_size + 10, 20), font, 0.5, (0, 255, 0), 1)
            cv2.putText(preview, "Enhanced", (target_size * 2 + 10, 20), font, 0.5, (0, 255, 0), 1)

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
    parser.add_argument("--clahe-clip", type=float, default=3.0, help="CLAHE clip limit (default 3.0)")
    parser.add_argument("--size", type=int, default=224, help="Output image size (default 224)")
    args = parser.parse_args()

    process_dataset(args.input, args.output, args.crop_ratio, args.clahe_clip, args.size)
