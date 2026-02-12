"""Auto-annotation API routes — triggers async Celery tasks."""

from fastapi import APIRouter

from app.schemas.annotation import (
    AutoAnnotateRequest,
    SAMPromptRequest,
    CLIPSearchRequest,
    GroundedSAMRequest,
    AutoAnnotateResponse,
)
from app.tasks.auto_annotate_tasks import (
    run_grounding_dino,
    run_sam2_predict,
    run_clip_search,
    run_grounded_sam,
    build_clip_index,
)

router = APIRouter()


@router.post("/grounding-dino", response_model=AutoAnnotateResponse)
async def auto_annotate_grounding_dino(request: AutoAnnotateRequest):
    """
    Text-prompted object detection using Grounding DINO.

    Flow:
    1. User types class names separated by periods (e.g., "hard hat . person . vest .")
    2. Grounding DINO's BERT encoder tokenizes the text prompt
    3. Swin Transformer extracts multi-scale image features
    4. Bidirectional cross-attention fuses text + image features
    5. Top-900 proposals are selected, NMS filters duplicates
    6. Returns bounding boxes with class labels and confidence scores
    """
    task = run_grounding_dino.delay(
        image_ids=[str(id) for id in request.image_ids],
        text_prompt=request.text_prompt,
        box_threshold=request.box_threshold,
        text_threshold=request.text_threshold,
    )
    return AutoAnnotateResponse(
        task_id=task.id,
        status="queued",
        message=f"Grounding DINO auto-annotation queued for {len(request.image_ids)} images",
    )


@router.post("/sam2", response_model=AutoAnnotateResponse)
async def auto_annotate_sam2(request: SAMPromptRequest):
    """
    SAM 2 mask generation from box or point prompts.

    Flow:
    1. Image Encoder (Hiera ViT) runs ONCE per image — cached for subsequent prompts
    2. Prompt Encoder converts box/point inputs to positional embeddings
    3. Mask Decoder generates 3 mask candidates + confidence scores
    4. Best mask is selected, returned as RLE-encoded binary mask
    5. Subsequent prompts on same image skip encoding (~8ms per prompt)
    """
    task = run_sam2_predict.delay(
        image_id=str(request.image_id),
        box_prompts=[b.model_dump() for b in request.box_prompts] if request.box_prompts else None,
        point_prompts=request.point_prompts,
    )
    return AutoAnnotateResponse(
        task_id=task.id,
        status="queued",
        message="SAM2 mask generation queued",
    )


@router.post("/clip-search", response_model=AutoAnnotateResponse)
async def clip_visual_search(request: CLIPSearchRequest):
    """
    CLIP-based visual similarity search (like Roboflow Rapid).

    Flow:
    1. User draws a bounding box on an object of interest
    2. The cropped region is passed through CLIP ViT-L/14 → 768-dim vector
    3. FAISS index (pre-built) searches for K nearest neighbors by cosine similarity
    4. Returns matching image regions sorted by similarity score
    5. User can accept matches to auto-annotate similar objects across dataset
    """
    task = run_clip_search.delay(
        image_id=str(request.image_id),
        crop_bbox=request.crop_bbox.model_dump(),
        top_k=request.top_k,
        threshold=request.threshold,
    )
    return AutoAnnotateResponse(
        task_id=task.id,
        status="queued",
        message=f"CLIP similarity search queued (top-{request.top_k})",
    )


@router.post("/grounded-sam", response_model=AutoAnnotateResponse)
async def auto_annotate_grounded_sam(request: GroundedSAMRequest):
    """
    Combined Grounding DINO + SAM 2 pipeline (the production power move).

    Flow:
    1. Grounding DINO detects objects from text prompt → bounding boxes
    2. Each box feeds into SAM 2 as a box prompt
    3. SAM 2 generates pixel-precise segmentation masks for each detection
    4. Returns complete annotations: boxes + masks + class labels
    5. Runs as batch Celery task on GPU workers
    """
    task = run_grounded_sam.delay(
        image_ids=[str(id) for id in request.image_ids],
        text_prompt=request.text_prompt,
        box_threshold=request.box_threshold,
        text_threshold=request.text_threshold,
        generate_masks=request.generate_masks,
    )
    return AutoAnnotateResponse(
        task_id=task.id,
        status="queued",
        message=f"Grounded SAM pipeline queued for {len(request.image_ids)} images",
    )


@router.post("/build-index/{project_id}", response_model=AutoAnnotateResponse)
async def build_project_clip_index(project_id: str):
    """
    Build FAISS index for a project's images using CLIP embeddings.
    Must be run before CLIP similarity search can work.
    """
    task = build_clip_index.delay(project_id=project_id)
    return AutoAnnotateResponse(
        task_id=task.id,
        status="queued",
        message="CLIP FAISS index build queued",
    )


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Check the status of an async auto-annotation task."""
    from app.tasks.celery_app import celery_app
    result = celery_app.AsyncResult(task_id)
    response = {
        "task_id": task_id,
        "status": result.status,
    }
    if result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = str(result.result)
    return response
