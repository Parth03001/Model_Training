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
_current_image_size = None  # Cache image dimensions to avoid re-opening


def _load_model():
    """Load SAM 2 model — called once per Celery worker."""
    global _predictor

    if _predictor is not None:
        return

    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from app.config import settings

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use local model folder if available, otherwise download from HuggingFace
        local_model_path = settings.model_base_dir / "sam2-hiera-large"
        model_id = str(local_model_path) if local_model_path.exists() else settings.sam2_model

        if local_model_path.exists():
            logger.info(f"Loading SAM 2 from local folder: {local_model_path}")
        else:
            logger.info(f"Local SAM2 not found at {local_model_path}, loading from HuggingFace: {model_id}")

        _predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
        logger.info(f"SAM 2 loaded on {device}")

    except ImportError:
        logger.warning("SAM 2 not installed. Using fallback stub.")
        _predictor = "stub"


def _set_image(image_path: str):
    """Set image for prediction — caches encoding for repeated prompts."""
    global _current_image_path, _current_image_size

    if _current_image_path == image_path:
        return  # Already encoded

    image = PILImage.open(image_path).convert("RGB")
    _current_image_size = image.size  # (width, height)
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

    # Use cached image size instead of re-opening the file
    width, height = _current_image_size

    # Convert all boxes to absolute xyxy format
    input_boxes = np.array([
        [
            int((b["x"] - b["w"] / 2) * width),
            int((b["y"] - b["h"] / 2) * height),
            int((b["x"] + b["w"] / 2) * width),
            int((b["y"] + b["h"] / 2) * height),
        ]
        for b in boxes
    ])

    # Predict masks — image encoding is cached, only the lightweight
    # mask decoder runs per box (~8ms each)
    results = []
    for i, box in enumerate(boxes):
        masks, scores, _ = _predictor.predict(
            box=input_boxes[i],
            multimask_output=True,
        )

        best_idx = np.argmax(scores)
        results.append({
            "mask_rle": _mask_to_rle(masks[best_idx]),
            "iou_score": float(scores[best_idx]),
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

    # Use cached image size
    width, height = _current_image_size

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
    """Convert binary mask to Run-Length Encoding (RLE) using vectorized numpy."""
    pixels = mask.flatten()

    # Vectorized RLE: find positions where values change
    diffs = np.diff(pixels)
    change_indices = np.where(diffs != 0)[0] + 1
    run_starts = np.concatenate([[0], change_indices])
    run_lengths = np.diff(np.concatenate([run_starts, [len(pixels)]]))

    return {
        "counts": run_lengths.tolist(),
        "size": list(mask.shape),
    }
