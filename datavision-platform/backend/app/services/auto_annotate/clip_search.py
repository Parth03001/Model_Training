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
_is_transformers = False  # Track if we're using transformers or open_clip

# In-memory FAISS index cache: {project_id: (index, metadata, mtime)}
_index_cache: dict[str, tuple] = {}


def _load_model():
    """Load CLIP/SigLIP model — called once per Celery worker."""
    global _model, _preprocess, _device, _tokenizer, _is_transformers

    if _model is not None:
        return

    from app.config import settings
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Using open_clip for flexibility
    # High-VRAM Config: Using SigLIP SO400M (384px) - SOTA for Retrieval
    local_siglip_folder = settings.model_base_dir / "siglip-so400m-patch14-384"

    if local_siglip_folder.exists():
        logger.info(f"Loading SigLIP model from local folder on {_device}: {local_siglip_folder}")
        try:
            from transformers import SiglipModel, SiglipImageProcessor
            _model = SiglipModel.from_pretrained(str(local_siglip_folder.resolve())).to(_device)
            _preprocess = SiglipImageProcessor.from_pretrained(str(local_siglip_folder.resolve()))
            _is_transformers = True
            logger.info("SigLIP loaded using transformers")
        except Exception as e:
            logger.error(f"Failed to load local SigLIP via transformers: {e}. Falling back...")
            _model = None

    if _model is None:
        import open_clip
        logger.warning(f"SigLIP folder not found or failed to load. Falling back to ViT-L-14 download on {_device}...")
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            device=_device,
        )
        # Cache tokenizer once (only for open_clip fallback)
        _tokenizer = open_clip.get_tokenizer("ViT-L-14")
        _is_transformers = False
        logger.info(f"CLIP ViT-L-14 loaded on {_device}")

    _model.eval()


def encode_image(image: PILImage.Image) -> np.ndarray:
    """Encode a PIL image to a 768-dim CLIP embedding."""
    _load_model()

    if _is_transformers:
        inputs = _preprocess(images=image, return_tensors="pt").to(_device)
        with torch.no_grad():
            outputs = _model.get_image_features(**inputs)
            # transformers returns a BaseModelOutputWithPooling object or similar
            features = outputs.pooler_output if hasattr(outputs, "pooler_output") else (outputs[0] if isinstance(outputs, (list, tuple)) else outputs)
            features = features / features.norm(dim=-1, keepdim=True)
    else:
        image_tensor = _preprocess(image).unsqueeze(0).to(_device)
        with torch.no_grad():
            features = _model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten()


def encode_image_batch(images: list[PILImage.Image], batch_size: int = 16) -> np.ndarray:
    """Encode a batch of PIL images to CLIP embeddings. Much faster than one-by-one."""
    _load_model()

    all_embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        
        if _is_transformers:
            inputs = _preprocess(images=batch, return_tensors="pt").to(_device)
            with torch.no_grad():
                outputs = _model.get_image_features(**inputs)
                # transformers returns a BaseModelOutputWithPooling object or similar
                features = outputs.pooler_output if hasattr(outputs, "pooler_output") else (outputs[0] if isinstance(outputs, (list, tuple)) else outputs)
                features = features / features.norm(dim=-1, keepdim=True)
        else:
            tensors = torch.stack([_preprocess(img) for img in batch]).to(_device)
            with torch.no_grad():
                features = _model.encode_image(tensors)
                features = features / features.norm(dim=-1, keepdim=True)

        all_embeddings.append(features.cpu().numpy())

    return np.vstack(all_embeddings)


def encode_text(text: str) -> np.ndarray:
    """Encode text to CLIP embedding (for text-to-image search)."""
    _load_model()

    if _is_transformers:
        # Note: SigLIP text encoding requires the tokenizer which had issues loading locally.
        # This feature is rarely used in the current index building flow.
        from transformers import AutoTokenizer
        try:
            local_siglip_folder = Path(_model.config._name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(str(local_siglip_folder))
            inputs = tokenizer([text], padding=True, return_tensors="pt").to(_device)
            with torch.no_grad():
                outputs = _model.get_text_features(**inputs)
                # transformers returns a BaseModelOutputWithPooling object or similar
                features = outputs.pooler_output if hasattr(outputs, "pooler_output") else (outputs[0] if isinstance(outputs, (list, tuple)) else outputs)
                features = features / features.norm(dim=-1, keepdim=True)
        except Exception as e:
            logger.error(f"Text encoding failed for SigLIP: {e}")
            return np.zeros(768)
    else:
        text_tokens = _tokenizer([text]).to(_device)
        with torch.no_grad():
            features = _model.encode_text(text_tokens)
            features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten()


def build_faiss_index(
    project_id: str,
    image_records: list[dict],
    progress_callback: callable = None,
) -> str:
    """
    Build FAISS index for a project's images with multi-scale support and progress reporting.
    """
    import faiss
    from app.config import settings

    _load_model()

    all_images = []   # PIL images to encode
    metadata = []     # Parallel metadata list

    total_images = len(image_records)
    for i, record in enumerate(image_records):
        if progress_callback:
            # First 30% of progress is for loading and cropping
            progress_callback(int((i / total_images) * 30))
            
        logger.info(f"[CLIP index] Loading image {i + 1}/{total_images}: {record['filepath']}")
        try:
            image = PILImage.open(record["filepath"]).convert("RGB")
            w, h = image.size

            # Multi-scale grid (1x1, 2x2, 4x4)
            # We use 1x1 (global) and 2x2, 4x4 for parts.
            scales = [1, 2, 4]
            
            for scale in scales:
                cell_w = w // scale
                cell_h = h // scale
                
                # Reduce overlap to speed up (stride by 0.75 of cell size instead of 0.5)
                step_x = int(cell_w * 0.75) if scale > 1 else cell_w
                step_y = int(cell_h * 0.75) if scale > 1 else cell_h
                
                # Ensure at least 1 step
                step_x = max(1, step_x)
                step_y = max(1, step_y)

                for y in range(0, h - cell_h + 1, step_y):
                    for x in range(0, w - cell_w + 1, step_x):
                        x1, y1 = x, y
                        x2, y2 = x + cell_w, y + cell_h
                        
                        # Skip if too small for CLIP/SigLIP (effectively < 64px)
                        if (x2 - x1) < 64 or (y2 - y1) < 64:
                            continue

                        crop = image.crop((x1, y1, x2, y2))
                        all_images.append(crop)
                        metadata.append({
                            "image_id": record["id"],
                            "bbox": [
                                (x1 + x2) / (2 * w),
                                (y1 + y2) / (2 * h),
                                (x2 - x1) / w,
                                (y2 - y1) / h,
                            ],
                            "type": f"grid_{scale}x{scale}",
                        })
        except Exception as e:
            logger.error(f"CLIP indexing failed for {record['filepath']}: {e}")

    if not all_images:
        raise ValueError("No images to index")

    # Batch encode all images at once
    logger.info(f"Encoding {len(all_images)} image crops in batches...")
    
    # Second part of progress (30% to 95%) is for encoding
    batch_size = 32
    vectors_list = []
    for b_idx in range(0, len(all_images), batch_size):
        batch = all_images[b_idx : b_idx + batch_size]
        vecs = encode_image_batch(batch, batch_size=batch_size)
        vectors_list.append(vecs)
        
        if progress_callback:
            p = 30 + int(((b_idx + len(batch)) / len(all_images)) * 65)
            progress_callback(p)

    vectors = np.vstack(vectors_list).astype(np.float32)

    # Build FAISS index
    if progress_callback: progress_callback(95)
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
    top_k: int = 5,
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

    # Apply simple Non-Maximum Suppression (NMS) to remove overlapping results for the same image
    filtered_results = []
    
    def get_iou(boxA, boxB):
        # box: [cx, cy, w, h]
        ax1, ay1 = boxA[0] - boxA[2]/2, boxA[1] - boxA[3]/2
        ax2, ay2 = boxA[0] + boxA[2]/2, boxA[1] + boxA[3]/2
        bx1, by1 = boxB[0] - boxB[2]/2, boxB[1] - boxB[3]/2
        bx2, by2 = boxB[0] + boxB[2]/2, boxB[1] + boxB[3]/2
        
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        areaA = boxA[2] * boxA[3]
        areaB = boxB[2] * boxB[3]
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0

    # Sort by similarity descending (already mostly sorted by FAISS but safety first)
    results.sort(key=lambda x: x["similarity"], reverse=True)

    for res in results:
        keep = True
        for existing in filtered_results:
            if res["image_id"] == existing["image_id"]:
                if get_iou(res["bbox"], existing["bbox"]) > 0.3: # 30% overlap threshold
                    keep = False
                    break
        if keep:
            filtered_results.append(res)

    logger.info(f"CLIP search: {len(filtered_results)} results after NMS (threshold {threshold})")
    return filtered_results
