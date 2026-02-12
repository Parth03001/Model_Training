"""Image processing utilities."""

import cv2
import numpy as np
from PIL import Image as PILImage
from pathlib import Path


def apply_clahe(image_path: str, clip_limit: float = 3.0, tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l_channel)

    enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def center_crop(image: PILImage.Image, crop_ratio: float = 0.55) -> PILImage.Image:
    """Crop the center of an image."""
    w, h = image.size
    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h))


def generate_thumbnail(image_path: str, output_path: str, size: tuple = (256, 256)):
    """Generate a thumbnail for an image."""
    img = PILImage.open(image_path)
    img.thumbnail(size, PILImage.Resampling.LANCZOS)
    img.save(output_path)


def rle_to_mask(rle: dict) -> np.ndarray:
    """Convert RLE encoding back to binary mask."""
    h, w = rle["size"]
    counts = rle["counts"]
    mask = np.zeros(h * w, dtype=np.uint8)

    pos = 0
    val = 0
    for count in counts:
        mask[pos:pos + count] = val
        pos += count
        val = 1 - val

    return mask.reshape(h, w)


def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    """Convert binary mask to polygon points."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Take the largest contour
    largest = max(contours, key=cv2.contourArea)
    h, w = mask.shape

    # Normalize to [0, 1]
    points = []
    for point in largest.squeeze():
        points.append([float(point[0]) / w, float(point[1]) / h])

    return points
