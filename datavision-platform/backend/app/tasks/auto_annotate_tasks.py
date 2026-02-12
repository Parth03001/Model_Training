"""Celery tasks for auto-annotation — run on GPU workers."""

import json
from loguru import logger

from app.tasks.celery_app import celery_app


@celery_app.task(name="app.tasks.auto_annotate_tasks.run_grounding_dino", bind=True)
def run_grounding_dino(self, image_ids: list[str], text_prompt: str, box_threshold: float, text_threshold: float):
    """
    Run Grounding DINO auto-annotation on a batch of images.
    Saves annotations to the database.
    """
    from app.services.auto_annotate.grounding_dino import batch_detect
    from app.database import async_session
    from app.models.image import Image
    from app.models.annotation import Annotation
    import asyncio
    from sqlalchemy import select

    async def _process():
        async with async_session() as db:
            # Fetch image records
            image_paths = {}
            for img_id in image_ids:
                result = await db.execute(select(Image).where(Image.id == img_id))
                img = result.scalar_one_or_none()
                if img:
                    image_paths[img_id] = img.filepath

            # Run detection
            all_detections = batch_detect(
                list(image_paths.values()),
                text_prompt,
                box_threshold,
                text_threshold,
            )

            # Save to database
            total_annotations = 0
            for img_id, filepath in image_paths.items():
                detections = all_detections.get(filepath, [])
                for det in detections:
                    ann = Annotation(
                        image_id=img_id,
                        class_name=det["class"],
                        annotation_type="bbox",
                        bbox_x=det["bbox"][0],
                        bbox_y=det["bbox"][1],
                        bbox_w=det["bbox"][2],
                        bbox_h=det["bbox"][3],
                        confidence=det["confidence"],
                        source="grounding_dino",
                    )
                    db.add(ann)
                    total_annotations += 1

                # Update image status
                result = await db.execute(select(Image).where(Image.id == img_id))
                img = result.scalar_one_or_none()
                if img:
                    img.status = "annotated"

            await db.commit()

        return {"total_annotations": total_annotations, "images_processed": len(image_ids)}

    return asyncio.get_event_loop().run_until_complete(_process())


@celery_app.task(name="app.tasks.auto_annotate_tasks.run_sam2_predict", bind=True)
def run_sam2_predict(self, image_id: str, box_prompts: list | None, point_prompts: list | None):
    """Run SAM 2 mask generation from box/point prompts."""
    from app.services.auto_annotate.sam2 import predict_from_boxes, predict_from_points
    from app.database import async_session
    from app.models.image import Image
    from app.models.annotation import Annotation
    import asyncio
    from sqlalchemy import select

    async def _process():
        async with async_session() as db:
            result = await db.execute(select(Image).where(Image.id == image_id))
            img = result.scalar_one_or_none()
            if not img:
                return {"error": "Image not found"}

            masks = []
            if box_prompts:
                masks = predict_from_boxes(img.filepath, box_prompts)
            elif point_prompts:
                masks = predict_from_points(img.filepath, point_prompts)

            # Save mask annotations
            for mask_result in masks:
                ann = Annotation(
                    image_id=image_id,
                    class_name="object",
                    annotation_type="mask",
                    mask_rle=json.dumps(mask_result.get("mask_rle", {})),
                    confidence=mask_result.get("iou_score"),
                    source="sam2",
                )
                if mask_result.get("bbox"):
                    ann.bbox_x = mask_result["bbox"]["x"]
                    ann.bbox_y = mask_result["bbox"]["y"]
                    ann.bbox_w = mask_result["bbox"]["w"]
                    ann.bbox_h = mask_result["bbox"]["h"]
                db.add(ann)

            await db.commit()

        return {"masks_generated": len(masks)}

    return asyncio.get_event_loop().run_until_complete(_process())


@celery_app.task(name="app.tasks.auto_annotate_tasks.run_clip_search", bind=True)
def run_clip_search(self, image_id: str, crop_bbox: dict, top_k: int, threshold: float):
    """Run CLIP visual similarity search."""
    from app.services.auto_annotate.clip_search import search_similar
    from app.database import async_session
    from app.models.image import Image
    import asyncio
    from sqlalchemy import select

    async def _process():
        async with async_session() as db:
            result = await db.execute(select(Image).where(Image.id == image_id))
            img = result.scalar_one_or_none()
            if not img:
                return {"error": "Image not found"}

            results = search_similar(
                str(img.project_id),
                img.filepath,
                crop_bbox,
                top_k,
                threshold,
            )
            return {"results": results, "total": len(results)}

    return asyncio.get_event_loop().run_until_complete(_process())


@celery_app.task(name="app.tasks.auto_annotate_tasks.run_grounded_sam", bind=True)
def run_grounded_sam(
    self, image_ids: list[str], text_prompt: str,
    box_threshold: float, text_threshold: float, generate_masks: bool,
):
    """Run combined Grounding DINO + SAM 2 pipeline."""
    from app.services.auto_annotate.grounded_sam import batch_annotate
    from app.database import async_session
    from app.models.image import Image
    from app.models.annotation import Annotation
    import asyncio
    from sqlalchemy import select

    async def _process():
        async with async_session() as db:
            image_paths = {}
            for img_id in image_ids:
                result = await db.execute(select(Image).where(Image.id == img_id))
                img = result.scalar_one_or_none()
                if img:
                    image_paths[img_id] = img.filepath

            all_results = batch_annotate(
                list(image_paths.values()),
                text_prompt, box_threshold, text_threshold, generate_masks,
            )

            total = 0
            for img_id, filepath in image_paths.items():
                annotations = all_results.get(filepath, [])
                for ann_data in annotations:
                    ann = Annotation(
                        image_id=img_id,
                        class_name=ann_data["class"],
                        annotation_type="mask" if ann_data.get("mask_rle") else "bbox",
                        bbox_x=ann_data["bbox"][0],
                        bbox_y=ann_data["bbox"][1],
                        bbox_w=ann_data["bbox"][2],
                        bbox_h=ann_data["bbox"][3],
                        confidence=ann_data["confidence"],
                        mask_rle=json.dumps(ann_data["mask_rle"]) if ann_data.get("mask_rle") else None,
                        source="grounded_sam",
                    )
                    db.add(ann)
                    total += 1

                result = await db.execute(select(Image).where(Image.id == img_id))
                img = result.scalar_one_or_none()
                if img:
                    img.status = "annotated"

            await db.commit()

        return {"total_annotations": total, "images_processed": len(image_ids)}

    return asyncio.get_event_loop().run_until_complete(_process())


@celery_app.task(name="app.tasks.auto_annotate_tasks.build_clip_index", bind=True)
def build_clip_index(self, project_id: str):
    """Build FAISS index for CLIP similarity search."""
    from app.services.auto_annotate.clip_search import build_faiss_index
    from app.database import async_session
    from app.models.image import Image
    import asyncio
    from sqlalchemy import select

    async def _get_images():
        async with async_session() as db:
            result = await db.execute(select(Image).where(Image.project_id == project_id))
            images = result.scalars().all()
            return [
                {"id": str(img.id), "filepath": img.filepath, "width": img.width, "height": img.height}
                for img in images
            ]

    images = asyncio.get_event_loop().run_until_complete(_get_images())
    index_path = build_faiss_index(project_id, images)
    return {"index_path": index_path, "images_indexed": len(images)}
