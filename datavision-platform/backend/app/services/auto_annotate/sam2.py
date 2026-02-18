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
        from sam2.build_sam2 import build_sam2
        from app.config import settings
        import os

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use local model folder if available
        local_model_path = settings.model_base_dir / "sam2-hiera-large"
        
        if local_model_path.exists():
            checkpoint = os.path.abspath(str(local_model_path / "sam2_hiera_large.pt"))
            model_cfg = os.path.abspath(str(local_model_path / "sam2_hiera_l.yaml"))
            
            if not os.path.exists(checkpoint) or not os.path.exists(model_cfg):
                raise FileNotFoundError(f"Missing config or checkpoint in {local_model_path}")

            logger.info(f"Building SAM 2 from local checkpoint: {checkpoint}")
            # Some versions of SAM2 require the config file name, others the full path.
            # We'll try to build it directly.
            sam2_model = build_sam2(model_cfg, checkpoint, device=device)
            _predictor = SAM2ImagePredictor(sam2_model)
            logger.info("SAM 2 loaded successfully")
        else:
            logger.warning("Local SAM2 checkpoint not found.")
            _predictor = "stub"

    except Exception as e:
        logger.error(f"Failed to load SAM 2: {e}")
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

    # Also generate center points to help SAM2 focus on the object vs background
    input_points = np.array([
        [int(b["x"] * width), int(b["y"] * height)]
        for b in boxes
    ])
    input_labels = np.ones(len(boxes)) # All foreground

    # Predict masks — image encoding is cached, only the lightweight
    # mask decoder runs per box (~8ms each)
    results = []
    for i, box in enumerate(boxes):
        # Using both box and point prompt is the "modern" way to get high precision
        masks, scores, _ = _predictor.predict(
            point_coords=input_points[i:i+1],
            point_labels=input_labels[i:i+1],
            box=input_boxes[i],
            multimask_output=True,
        )

        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        # Calculate tight bbox from the actual mask pixels
        refined_bbox = _get_tight_bbox(best_mask, width, height) or box

        results.append({
            "mask_rle": _mask_to_rle(best_mask),
            "iou_score": float(scores[best_idx]),
            "bbox": refined_bbox,
        })

    logger.info(f"SAM 2: Generated {len(results)} masks and refined boxes")
    return results


def _get_tight_bbox(mask: np.ndarray, width: int, height: int) -> dict | None:
    """Calculate perfect [cx, cy, w, h] by finding the largest connected component's bbox."""
    from scipy.ndimage import label
    
    # Filter small noise fragments that make boxes too large
    structure = np.ones((3, 3), dtype=int)
    labeled, n_components = label(mask, structure=structure)
    
    if n_components == 0:
        return None
        
    # Find the largest component (usually the actual object)
    component_sizes = np.bincount(labeled.ravel())
    largest_idx = component_sizes[1:].argmax() + 1
    tight_mask = (labeled == largest_idx)
    
    rows = np.any(tight_mask, axis=1)
    cols = np.any(tight_mask, axis=0)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # 0.5px offset for sub-pixel precision
    bw = (cmax - cmin + 1) / width
    bh = (rmax - rmin + 1) / height
    cx = (cmin + cmax + 1) / (2 * width)
    cy = (rmin + rmax + 1) / (2 * height)
    
    return {"x": cx, "y": cy, "w": bw, "h": bh}


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
    best_mask = masks[best_idx]
    refined_bbox = _get_tight_bbox(best_mask, width, height)

    return [{
        "mask_rle": _mask_to_rle(best_mask),
        "iou_score": float(scores[best_idx]),
        "bbox": refined_bbox
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
