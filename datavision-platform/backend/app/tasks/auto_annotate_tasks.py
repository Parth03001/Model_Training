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

    return asyncio.run(_process())


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

    return asyncio.run(_process())


@celery_app.task(name="app.tasks.auto_annotate_tasks.run_clip_search", bind=True)
def run_clip_search(self, image_id: str, crop_bbox: dict, class_name: str | None, top_k: int, threshold: float):
    """Run CLIP visual similarity search. Auto-builds FAISS index if not found."""
    from app.services.auto_annotate.clip_search import search_similar, build_faiss_index
    from app.database import async_session
    from app.models.image import Image
    from app.models.annotation import Annotation
    import asyncio
    from sqlalchemy import select

    async def _get_image_info():
        async with async_session() as db:
            result = await db.execute(select(Image).where(Image.id == image_id))
            img = result.scalar_one_or_none()
            if not img:
                return None, None
            
            # If class_name missing, try to find matching annotation at this bbox
            final_class = class_name
            if not final_class:
                ann_result = await db.execute(
                    select(Annotation).where(
                        Annotation.image_id == image_id,
                        Annotation.bbox_x == crop_bbox["x"],
                        Annotation.bbox_y == crop_bbox["y"]
                    )
                )
                ann = ann_result.scalar_one_or_none()
                if ann:
                    final_class = ann.class_name
                else:
                    final_class = "similar_object" # Fallback

            return img.project_id, img.filepath, final_class

    async def _get_all_images(project_id):
        async with async_session() as db:
            result = await db.execute(select(Image).where(Image.project_id == project_id))
            images = result.scalars().all()
            return [
                {"id": str(i.id), "filepath": i.filepath, "width": i.width, "height": i.height}
                for i in images
            ]

    async def _save_annotations(results, class_name):
        async with async_session() as db:
            from app.services.auto_annotate.sam2 import predict_from_boxes
            import os

            saved_count = 0
            # Group results by image_id to batch SAM 2 processing
            by_image = {}
            for res in results:
                # Skip the exact source region
                if res["image_id"] == image_id and \
                   abs(res["bbox"][0] - crop_bbox["x"]) < 0.01 and \
                   abs(res["bbox"][1] - crop_bbox["y"]) < 0.01:
                    continue
                
                if res["image_id"] not in by_image:
                    by_image[res["image_id"]] = []
                by_image[res["image_id"]].append(res)

            for img_id, img_results in by_image.items():
                # Fetch image path for SAM 2
                res_img = await db.execute(select(Image).where(Image.id == img_id))
                img_record = res_img.scalar_one_or_none()
                if not img_record or not os.path.exists(img_record.filepath):
                    continue

                # Refine boxes with SAM 2 for "perfection"
                try:
                    # For small objects like bolts, a single center point might hit a shadow.
                    # We'll use a 3x3 point grid inside the box as prompts to force SAM2
                    # to see the whole object clearly.
                    refined_results = []
                    for r in img_results:
                        box = {"x": r["bbox"][0], "y": r["bbox"][1], "w": r["bbox"][2], "h": r["bbox"][3]}
                        
                        # Generate 9 points: 
                        # 5 Positives: Center + 4 Inset Corners
                        # 4 Negatives: Points just outside the box to constrain it
                        pad_w, pad_h = box["w"] * 0.1, box["h"] * 0.1
                        points = [
                            {"x": box["x"], "y": box["y"], "label": 1}, # Center
                            {"x": box["x"] - box["w"]*0.2, "y": box["y"] - box["h"]*0.2, "label": 1},
                            {"x": box["x"] + box["w"]*0.2, "y": box["y"] - box["h"]*0.2, "label": 1},
                            {"x": box["x"] - box["w"]*0.2, "y": box["y"] + box["h"]*0.2, "label": 1},
                            {"x": box["x"] + box["w"]*0.2, "y": box["y"] + box["h"]*0.2, "label": 1},
                            
                            # Negative points (Background) to prevent "Box Bloat"
                            {"x": box["x"] - box["w"]*0.6, "y": box["y"], "label": 0},
                            {"x": box["x"] + box["w"]*0.6, "y": box["y"], "label": 0},
                            {"x": box["x"], "y": box["y"] - box["h"]*0.6, "label": 0},
                            {"x": box["x"], "y": box["y"] + box["h"]*0.6, "label": 0},
                        ]
                        
                        from app.services.auto_annotate.sam2 import predict_from_points
                        # Using multiple points inside the detection is the most robust way
                        sam_res = predict_from_points(img_record.filepath, points)
                        if sam_res and sam_res[0].get("bbox"):
                            refined_results.append(sam_res[0]["bbox"])
                        else:
                            refined_results.append(box)

                    for i, refined_bbox in enumerate(refined_results):
                        ann = Annotation(
                            image_id=img_id,
                            class_name=class_name,
                            annotation_type="bbox",
                            bbox_x=refined_bbox["x"],
                            bbox_y=refined_bbox["y"],
                            bbox_w=refined_bbox["w"],
                            bbox_h=refined_bbox["h"],
                            confidence=img_results[i]["similarity"],
                            source="clip_search",
                            is_verified=False
                        )
                        db.add(ann)
                        saved_count += 1
                except Exception as e:
                    logger.error(f"SAM 2 refinement failed for {img_id}: {e}")
                    # Fallback to unrefined boxes
                    for res in img_results:
                        ann = Annotation(
                            image_id=img_id,
                            class_name=class_name,
                            annotation_type="bbox",
                            bbox_x=res["bbox"][0],
                            bbox_y=res["bbox"][1],
                            bbox_w=res["bbox"][2],
                            bbox_h=res["bbox"][3],
                            confidence=res["similarity"],
                            source="clip_search",
                            is_verified=False
                        )
                        db.add(ann)
                        saved_count += 1

            await db.commit()
            return saved_count

    project_id, filepath, final_class = asyncio.run(_get_image_info())
    if not project_id:
        return {"error": "Image not found"}

    try:
        results = search_similar(str(project_id), filepath, crop_bbox, top_k, threshold)
    except FileNotFoundError:
        # FAISS index not built yet — build it automatically then search
        logger.info(f"FAISS index missing for project {project_id}. Auto-building...")
        images = asyncio.run(_get_all_images(project_id))
        if not images:
            return {"error": "No images in project to build index from"}
            
        def update_progress(current_percent):
            self.update_state(state='PROGRESS', meta={'progress': current_percent})
            
        build_faiss_index(str(project_id), images, progress_callback=update_progress)
        results = search_similar(str(project_id), filepath, crop_bbox, top_k, threshold)

    # Save findings as unverified annotations so they show up on images for "Accept/Reject"
    saved_count = asyncio.run(_save_annotations(results, final_class))

    return {
        "results": results, 
        "total": len(results), 
        "applied_count": saved_count,
        "class_name": final_class
    }


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
                    # Final refinement of Grounding DINO boxes using our "Perfect Box" logic
                    # This ensures DINO's rough boxes are tightened by SAM2
                    final_bbox = ann_data["bbox"]
                    if not ann_data.get("mask_rle"):
                        # If batch_annotate didn't already use SAM2 masks, we refine here
                        try:
                            from app.services.auto_annotate.sam2 import predict_from_boxes
                            refined = predict_from_boxes(filepath, [{"x": final_bbox[0], "y": final_bbox[1], "w": final_bbox[2], "h": final_bbox[3]}])
                            if refined and refined[0].get("bbox"):
                                rbox = refined[0]["bbox"]
                                final_bbox = [rbox["x"], rbox["y"], rbox["w"], rbox["h"]]
                        except:
                            pass

                    ann = Annotation(
                        image_id=img_id,
                        class_name=ann_data["class"],
                        annotation_type="mask" if ann_data.get("mask_rle") else "bbox",
                        bbox_x=final_bbox[0],
                        bbox_y=final_bbox[1],
                        bbox_w=final_bbox[2],
                        bbox_h=final_bbox[3],
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

    return asyncio.run(_process())


@celery_app.task(name="app.tasks.auto_annotate_tasks.build_clip_index", bind=True)
def build_clip_index(self, project_id: str):
    """Build FAISS index for CLIP similarity search with progress reporting."""
    from app.services.auto_annotate.clip_search import build_faiss_index
    from app.database import async_session
    from app.models.image import Image
    import asyncio
    from sqlalchemy import select

    def update_progress(current_percent):
        self.update_state(state='PROGRESS', meta={'progress': current_percent})

    async def _get_images():
        async with async_session() as db:
            result = await db.execute(select(Image).where(Image.project_id == project_id))
            images = result.scalars().all()
            return [
                {"id": str(img.id), "filepath": img.filepath, "width": img.width, "height": img.height}
                for img in images
            ]

    images = asyncio.run(_get_images())
    index_path = build_faiss_index(project_id, images, progress_callback=update_progress)
    return {"index_path": index_path, "images_indexed": len(images)}


@celery_app.task(name="app.tasks.auto_annotate_tasks.run_trained_model_inference", bind=True)
def run_trained_model_inference(self, project_id: str, confidence_threshold: float, image_ids: list[str] | None = None):
    """Run inference using the latest trained model for a project."""
    from app.database import async_session
    from app.models.image import Image
    from app.models.annotation import Annotation
    from app.models.trained_model import TrainedModel
    from ultralytics import YOLO
    import asyncio
    from sqlalchemy import select, func
    import os

    async def _process():
        async with async_session() as db:
            # 1. Find the latest best model for this project
            result = await db.execute(
                select(TrainedModel)
                .where(TrainedModel.project_id == project_id)
                .order_by(TrainedModel.created_at.desc())
                .limit(1)
            )
            model_record = result.scalar_one_or_none()
            if not model_record or not model_record.model_path or not os.path.exists(model_record.model_path):
                return {"error": "No trained model found for this project"}

            # 2. Get images to annotate
            if image_ids:
                img_query = select(Image).where(Image.id.in_(image_ids))
            else:
                # Find images in this project that have 0 annotations
                # Subquery to count annotations per image
                subq = select(Annotation.image_id, func.count(Annotation.id).label('count')).group_by(Annotation.image_id).subquery()
                img_query = (
                    select(Image)
                    .outerjoin(subq, Image.id == subq.c.image_id)
                    .where(Image.project_id == project_id)
                    .where(func.coalesce(subq.c.count, 0) == 0)
                )
            
            res_images = await db.execute(img_query)
            images_to_label = res_images.scalars().all()
            
            if not images_to_label:
                return {"message": "No images found that need auto-labeling"}

            # 3. Run YOLO inference
            logger.info(f"Loading custom model from {model_record.model_path}")
            model = YOLO(model_record.model_path)
            
            total_added = 0
            for idx, img in enumerate(images_to_label):
                self.update_state(state='PROGRESS', meta={'progress': int((idx / len(images_to_label)) * 100)})
                
                results = model.predict(img.filepath, conf=confidence_threshold, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Convert to normalized center
                        # YOLO returns [x1, y1, x2, y2] in pixels
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_idx = int(box.cls[0].cpu().numpy())
                        class_name = model.names[cls_idx]
                        
                        # Normalize
                        w, h = img.width, img.height
                        cx = ((xyxy[0] + xyxy[2]) / 2) / w
                        cy = ((xyxy[1] + xyxy[3]) / 2) / h
                        bw = (xyxy[2] - xyxy[0]) / w
                        bh = (xyxy[3] - xyxy[1]) / h
                        
                        ann = Annotation(
                            image_id=img.id,
                            class_name=class_name,
                            annotation_type="bbox",
                            bbox_x=float(cx),
                            bbox_y=float(cy),
                            bbox_w=float(bw),
                            bbox_h=float(bh),
                            confidence=conf,
                            source="custom_model",
                            is_verified=False
                        )
                        db.add(ann)
                        total_added += 1
                
                img.status = "annotated"
            
            await db.commit()
            return {"total_annotations": total_added, "images_processed": len(images_to_label)}

    return asyncio.run(_process())
