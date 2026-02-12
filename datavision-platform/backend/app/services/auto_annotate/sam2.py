"""
SAM 2 Service — Segment Anything Model 2 for mask generation.

Technical Flow:
    1. Image Encoder (Hiera ViT):
       - Hierarchical Vision Transformer processes the image ONCE
       - Produces multi-scale feature maps at 4 levels
       - This is the expensive step (~50ms on A100) — result is CACHED
       - Subsequent prompts on the same image skip this step entirely

    2. Prompt Encoder:
       - Converts user inputs to positional embeddings:
         * Box prompt → 2 corner points → sinusoidal positional encodings
         * Point prompt → single point + foreground/background label
         * Mask prompt → downsampled + convolution layers
       - Very lightweight (~1ms)

    3. Mask Decoder:
       - 2-layer transformer decoder with cross-attention to image features
       - Outputs 3 mask candidates at different granularity levels:
         * Whole object mask
         * Part-level mask
         * Sub-part mask
       - Each mask has an IoU prediction score
       - Best mask selected by highest predicted IoU

    4. Output:
       - Binary mask at original image resolution
       - Encoded as RLE (Run-Length Encoding) for efficient storage
       - RLE is 10-50× smaller than raw pixel mask

VRAM: ~3.5GB for hiera-large model
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
from loguru import logger

_predictor = None
_current_image_path = None  # Track cached image


def _load_model():
    """Load SAM 2 model — called once per Celery worker."""
    global _predictor

    if _predictor is not None:
        return

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from app.config import settings

        logger.info(f"Loading SAM 2: {settings.sam2_model}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        sam2_model = build_sam2(
            config_file="sam2_hiera_l.yaml",
            ckpt_path=settings.sam2_checkpoint,
            device=device,
        )
        _predictor = SAM2ImagePredictor(sam2_model)
        logger.info(f"SAM 2 loaded on {device}")

    except ImportError:
        logger.warning("SAM 2 not installed. Using fallback stub.")
        _predictor = "stub"


def _set_image(image_path: str):
    """Set image for prediction — caches encoding for repeated prompts."""
    global _current_image_path

    if _current_image_path == image_path:
        return  # Already encoded

    image = PILImage.open(image_path).convert("RGB")
    image_np = np.array(image)

    if _predictor != "stub":
        _predictor.set_image(image_np)

    _current_image_path = image_path
    logger.debug(f"SAM 2: Image encoded and cached for {image_path}")


def predict_from_boxes(
    image_path: str,
    boxes: list[dict],
) -> list[dict]:
    """
    Generate masks from bounding box prompts.

    Args:
        image_path: Path to image
        boxes: List of {"x": cx, "y": cy, "w": w, "h": h} in normalized [0-1] coords

    Returns:
        List of {"mask_rle": dict, "iou_score": float, "bbox": dict}
    """
    _load_model()

    if _predictor == "stub":
        return [{"mask_rle": {}, "iou_score": 0.9, "bbox": b} for b in boxes]

    _set_image(image_path)

    image = PILImage.open(image_path)
    width, height = image.size

    results = []
    for box in boxes:
        # Convert normalized center format to absolute xyxy
        cx, cy, w, h = box["x"], box["y"], box["w"], box["h"]
        x1 = int((cx - w / 2) * width)
        y1 = int((cy - h / 2) * height)
        x2 = int((cx + w / 2) * width)
        y2 = int((cy + h / 2) * height)

        input_box = np.array([x1, y1, x2, y2])

        masks, scores, _ = _predictor.predict(
            box=input_box,
            multimask_output=True,
        )

        # Select best mask (highest IoU score)
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = float(scores[best_idx])

        # Encode mask as RLE
        rle = _mask_to_rle(best_mask)

        results.append({
            "mask_rle": rle,
            "iou_score": best_score,
            "bbox": box,
        })

    logger.info(f"SAM 2: Generated {len(results)} masks from box prompts")
    return results


def predict_from_points(
    image_path: str,
    points: list[dict],
) -> list[dict]:
    """
    Generate masks from point prompts.

    Args:
        image_path: Path to image
        points: List of {"x": float, "y": float, "label": 1|0}
                label=1 for foreground, label=0 for background

    Returns:
        List of mask results
    """
    _load_model()

    if _predictor == "stub":
        return [{"mask_rle": {}, "iou_score": 0.9}]

    _set_image(image_path)

    image = PILImage.open(image_path)
    width, height = image.size

    point_coords = np.array([[int(p["x"] * width), int(p["y"] * height)] for p in points])
    point_labels = np.array([p.get("label", 1) for p in points])

    masks, scores, _ = _predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    best_idx = np.argmax(scores)
    return [{
        "mask_rle": _mask_to_rle(masks[best_idx]),
        "iou_score": float(scores[best_idx]),
    }]


def _mask_to_rle(mask: np.ndarray) -> dict:
    """Convert binary mask to Run-Length Encoding (RLE)."""
    pixels = mask.flatten()
    runs = []
    current_val = 0
    current_len = 0

    for pixel in pixels:
        if pixel == current_val:
            current_len += 1
        else:
            runs.append(current_len)
            current_val = pixel
            current_len = 1
    runs.append(current_len)

    return {
        "counts": runs,
        "size": list(mask.shape),
    }
