"""
CLIP Visual Similarity Search Service — Find similar objects across dataset.

Technical Flow:
    1. Index Building (one-time per dataset):
       a. Load CLIP ViT-L/14 model (produces 768-dim embeddings)
       b. For each image in the dataset:
          - Optionally run a sliding window / grid crop
          - Pass each crop through CLIP image encoder
          - Store 768-dim vector in FAISS index
       c. Save FAISS index to disk for fast loading
       d. Runs as background Celery task (~2 images/sec on GPU)

    2. Query (real-time):
       a. User draws a bounding box on an object in the annotation canvas
       b. Backend crops the region from the image
       c. CLIP encodes the crop → 768-dim query vector
       d. FAISS searches the pre-built index for K nearest neighbors
       e. Returns ranked list of (image_id, crop_bbox, similarity_score)
       f. Latency: ~10ms for search on 100K vectors

    3. FAISS Index Types:
       - IndexFlatIP: Brute-force inner product (exact, good for <100K vectors)
       - IndexIVFFlat: Inverted file with flat quantizer (faster for 100K-1M)
       - IndexHNSWFlat: Hierarchical Navigable Small World (best recall/speed)

VRAM: ~1.5GB for CLIP ViT-L/14
FAISS Memory: ~3MB per 1000 768-dim vectors
"""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
from loguru import logger

_model = None
_preprocess = None
_device = None
_tokenizer = None  # Cached tokenizer

# In-memory FAISS index cache: {project_id: (index, metadata, mtime)}
_index_cache: dict[str, tuple] = {}


def _load_model():
    """Load CLIP model — called once per Celery worker."""
    global _model, _preprocess, _device, _tokenizer

    if _model is not None:
        return

    import open_clip
    from app.config import settings

    logger.info(f"Loading CLIP model: {settings.clip_model}")

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    # Using open_clip for flexibility
    # High-VRAM Config: Using SigLIP SO400M (384px) - SOTA for Retrieval
    local_siglip_folder = Path("siglip-so400m-patch14-384")
    
    if local_siglip_folder.exists():
        logger.info(f"Loading SOTA SigLIP model from local folder: {local_siglip_folder}")
        # SigLIP requires loading via the 'hf-hub:' prefix pointing to the local folder
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "hf-hub:" + str(local_siglip_folder.resolve()),
            device=_device,
        )
    else:
        logger.warning("SigLIP folder not found! Falling back to standard ViT-L-14 download...")
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            device=_device,
        )

    _model.eval()

    # Cache tokenizer once
    _tokenizer = open_clip.get_tokenizer("ViT-L-14")

    logger.info(f"CLIP loaded on {_device}")


def encode_image(image: PILImage.Image) -> np.ndarray:
    """Encode a PIL image to a 768-dim CLIP embedding."""
    _load_model()

    image_tensor = _preprocess(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        features = _model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize

    return features.cpu().numpy().flatten()


def encode_image_batch(images: list[PILImage.Image], batch_size: int = 16) -> np.ndarray:
    """Encode a batch of PIL images to CLIP embeddings. Much faster than one-by-one."""
    _load_model()

    all_embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        tensors = torch.stack([_preprocess(img) for img in batch]).to(_device)

        with torch.no_grad():
            features = _model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)

        all_embeddings.append(features.cpu().numpy())

    return np.vstack(all_embeddings)


def encode_text(text: str) -> np.ndarray:
    """Encode text to CLIP embedding (for text-to-image search)."""
    _load_model()

    text_tokens = _tokenizer([text]).to(_device)

    with torch.no_grad():
        features = _model.encode_text(text_tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten()


def build_faiss_index(
    project_id: str,
    image_records: list[dict],
    grid_size: int = 3,
) -> str:
    """
    Build FAISS index for a project's images.

    Args:
        project_id: Project UUID
        image_records: List of {"id": str, "filepath": str, "width": int, "height": int}
        grid_size: Grid subdivision for each image (3 = 9 crops per image)

    Returns:
        Path to saved FAISS index
    """
    import faiss
    from app.config import settings

    _load_model()

    all_images = []   # PIL images to encode
    metadata = []     # Parallel metadata list

    for record in image_records:
        try:
            image = PILImage.open(record["filepath"]).convert("RGB")
            w, h = image.size

            # Full image
            all_images.append(image)
            metadata.append({
                "image_id": record["id"],
                "bbox": [0.5, 0.5, 1.0, 1.0],
                "type": "full",
            })

            # Grid crop embeddings
            cell_w = w // grid_size
            cell_h = h // grid_size
            for row in range(grid_size):
                for col in range(grid_size):
                    x1 = col * cell_w
                    y1 = row * cell_h
                    x2 = min(x1 + cell_w, w)
                    y2 = min(y1 + cell_h, h)

                    crop = image.crop((x1, y1, x2, y2))
                    if crop.size[0] < 32 or crop.size[1] < 32:
                        continue

                    all_images.append(crop)
                    metadata.append({
                        "image_id": record["id"],
                        "bbox": [
                            (x1 + x2) / (2 * w),
                            (y1 + y2) / (2 * h),
                            (x2 - x1) / w,
                            (y2 - y1) / h,
                        ],
                        "type": "grid",
                    })

        except Exception as e:
            logger.error(f"CLIP indexing failed for {record['filepath']}: {e}")

    if not all_images:
        raise ValueError("No images to index")

    # Batch encode all images at once (much faster than one-by-one)
    logger.info(f"Encoding {len(all_images)} image crops in batches...")
    vectors = encode_image_batch(all_images, batch_size=32).astype(np.float32)

    # Build FAISS index
    dim = vectors.shape[1]

    # Use inner product (cosine similarity since vectors are L2-normalized)
    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)
    ids = np.arange(len(vectors), dtype=np.int64)
    index.add_with_ids(vectors, ids)

    # Save index and metadata
    index_dir = Path(settings.faiss_index_dir) / project_id
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "clip.index"
    metadata_path = index_dir / "metadata.json"

    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    # Update in-memory cache
    _index_cache[project_id] = (index, metadata, index_path.stat().st_mtime)

    logger.info(f"FAISS index built: {len(vectors)} vectors, saved to {index_path}")
    return str(index_path)


def _load_index(project_id: str):
    """Load FAISS index and metadata, using in-memory cache when possible."""
    import faiss
    from app.config import settings

    index_dir = Path(settings.faiss_index_dir) / project_id
    index_path = index_dir / "clip.index"
    metadata_path = index_dir / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found for project {project_id}. Build it first.")

    current_mtime = index_path.stat().st_mtime

    # Return cached if still fresh
    if project_id in _index_cache:
        cached_index, cached_meta, cached_mtime = _index_cache[project_id]
        if cached_mtime == current_mtime:
            return cached_index, cached_meta

    # Load from disk and cache
    index = faiss.read_index(str(index_path))
    with open(metadata_path) as f:
        metadata = json.load(f)

    _index_cache[project_id] = (index, metadata, current_mtime)
    logger.debug(f"FAISS index loaded and cached for project {project_id}")
    return index, metadata


def search_similar(
    project_id: str,
    query_image_path: str,
    crop_bbox: dict,
    top_k: int = 20,
    threshold: float = 0.7,
) -> list[dict]:
    """
    Search for visually similar objects in the dataset.

    Args:
        project_id: Project UUID
        query_image_path: Path to the image containing the query object
        crop_bbox: {"x": cx, "y": cy, "w": w, "h": h} — region to crop
        top_k: Number of results to return
        threshold: Minimum similarity score

    Returns:
        List of {"image_id": str, "bbox": list, "similarity": float}
    """
    _load_model()

    # Load cached index
    index, metadata = _load_index(project_id)

    # Crop query region
    image = PILImage.open(query_image_path).convert("RGB")
    w, h = image.size

    cx, cy, bw, bh = crop_bbox["x"], crop_bbox["y"], crop_bbox["w"], crop_bbox["h"]
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    crop = image.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))

    # Encode query
    query_vec = encode_image(crop).reshape(1, -1).astype(np.float32)

    # Search
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or score < threshold:
            continue
        meta = metadata[idx]
        results.append({
            "image_id": meta["image_id"],
            "bbox": meta["bbox"],
            "similarity": float(score),
            "type": meta["type"],
        })

    logger.info(f"CLIP search: {len(results)} results above threshold {threshold}")
    return results
