"""
Grounded SAM Service — Combined Grounding DINO + SAM 2 pipeline.

Technical Flow:
    1. Grounding DINO processes text prompt → produces bounding boxes with class labels
    2. Each box is fed to SAM 2 as a box prompt
    3. SAM 2 generates a pixel-precise segmentation mask per box
    4. Output: Complete annotations with boxes + masks + class labels + confidence scores

    This is the "production power move" — one API call gives you everything.

Memory: ~6GB total (Grounding DINO ~2.5GB + SAM 2 ~3.5GB)
Speed: ~200ms per image on A100 (both models combined)
"""

from loguru import logger

from app.services.auto_annotate.grounding_dino import detect as dino_detect
from app.services.auto_annotate.sam2 import predict_from_boxes


def annotate_image(
    image_path: str,
    text_prompt: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    generate_masks: bool = True,
) -> list[dict]:
    """
    Run full Grounded SAM pipeline on a single image.

    Returns:
        List of annotations: [{
            "class": str,
            "bbox": [cx, cy, w, h],
            "confidence": float,
            "mask_rle": dict | None,
            "iou_score": float | None,
        }]
    """
    # Step 1: Grounding DINO detection
    detections = dino_detect(image_path, text_prompt, box_threshold, text_threshold)

    if not detections:
        logger.info(f"Grounded SAM: No detections for {image_path}")
        return []

    if not generate_masks:
        return detections

    # Step 2: SAM 2 mask generation for each detection
    boxes = [{"x": d["bbox"][0], "y": d["bbox"][1], "w": d["bbox"][2], "h": d["bbox"][3]} for d in detections]
    mask_results = predict_from_boxes(image_path, boxes)

    # Combine
    annotations = []
    for det, mask in zip(detections, mask_results):
        annotations.append({
            "class": det["class"],
            "bbox": det["bbox"],
            "confidence": det["confidence"],
            "mask_rle": mask["mask_rle"],
            "iou_score": mask["iou_score"],
        })

    logger.info(f"Grounded SAM: {len(annotations)} complete annotations for {image_path}")
    return annotations


def batch_annotate(
    image_paths: list[str],
    text_prompt: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    generate_masks: bool = True,
) -> dict[str, list[dict]]:
    """Run Grounded SAM on multiple images."""
    results = {}
    for path in image_paths:
        try:
            results[path] = annotate_image(path, text_prompt, box_threshold, text_threshold, generate_masks)
        except Exception as e:
            logger.error(f"Grounded SAM failed for {path}: {e}")
            results[path] = []
    return results
