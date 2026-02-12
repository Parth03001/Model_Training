"""
Grounding DINO Service — Text-prompted zero-shot object detection.

Technical Flow:
    1. Text Encoding: BERT tokenizer processes the text prompt. Periods (.) act as
       class separators — "hard hat . person . vest ." produces 3 class embeddings.
    2. Image Encoding: Swin Transformer backbone extracts multi-scale feature maps
       at 4 levels (stride 8, 16, 32, 64).
    3. Feature Enhancer: 6 layers of bidirectional cross-attention between text tokens
       and image features. This is the key innovation — text and image features
       "talk to each other" to ground language in visual regions.
    4. Language-Guided Query Selection: Top-900 image features most aligned with text
       are selected as object queries.
    5. Cross-Modality Decoder: 6 decoder layers refine queries into boxes + class logits.
    6. Post-processing: Box threshold filters low-confidence predictions, then NMS
       removes duplicates.

Model Variants:
    - grounding-dino-tiny (172M params, ~40ms/image on A100)
    - grounding-dino-base (341M params, ~60ms/image on A100)
    - Grounding DINO 1.5/1.6 (cloud API via IDEA Research)

VRAM: ~2.5GB for base model
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
from loguru import logger

# Singleton model instance (loaded once per worker process)
_model = None
_processor = None


def _load_model():
    """Load Grounding DINO model — called once per Celery worker."""
    global _model, _processor

    if _model is not None:
        return

    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from app.config import settings

    logger.info(f"Loading Grounding DINO: {settings.grounding_dino_model}")

    _processor = AutoProcessor.from_pretrained(settings.grounding_dino_model)
    _model = AutoModelForZeroShotObjectDetection.from_pretrained(settings.grounding_dino_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device)
    _model.eval()

    logger.info(f"Grounding DINO loaded on {device}")


def detect(
    image_path: str,
    text_prompt: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> list[dict]:
    """
    Run Grounding DINO detection on a single image.

    Args:
        image_path: Path to image file
        text_prompt: Class names separated by periods (e.g., "cat . dog . bird .")
        box_threshold: Minimum confidence for box predictions
        text_threshold: Minimum confidence for text-box matching

    Returns:
        List of detections: [{"class": str, "bbox": [cx, cy, w, h], "confidence": float}]
    """
    _load_model()
    device = next(_model.parameters()).device

    image = PILImage.open(image_path).convert("RGB")
    width, height = image.size

    # Ensure text prompt ends with a period
    if not text_prompt.strip().endswith("."):
        text_prompt = text_prompt.strip() + " ."

    # Process inputs
    inputs = _processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = _model(**inputs)

    # Post-process
    results = _processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(height, width)],
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = box.cpu().numpy()

        # Convert to normalized center format [cx, cy, w, h]
        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height

        detections.append({
            "class": label,
            "bbox": [float(cx), float(cy), float(w), float(h)],
            "confidence": float(score),
        })

    logger.info(f"Grounding DINO: {len(detections)} detections in {image_path}")
    return detections


def batch_detect(
    image_paths: list[str],
    text_prompt: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> dict[str, list[dict]]:
    """Run detection on multiple images."""
    results = {}
    for path in image_paths:
        try:
            results[path] = detect(path, text_prompt, box_threshold, text_threshold)
        except Exception as e:
            logger.error(f"Grounding DINO failed for {path}: {e}")
            results[path] = []
    return results
