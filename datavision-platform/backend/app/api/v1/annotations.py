"""Annotation CRUD API routes."""

import json
import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import select, func

from app.api.deps import DbSession
from app.models.annotation import Annotation
from app.models.image import Image
from app.schemas.annotation import (
    AnnotationCreate,
    AnnotationUpdate,
    AnnotationResponse,
)

router = APIRouter()


def _annotation_to_response(ann: Annotation) -> AnnotationResponse:
    bbox = None
    if ann.bbox_x is not None:
        from app.schemas.annotation import BBox
        bbox = BBox(x=ann.bbox_x, y=ann.bbox_y, w=ann.bbox_w, h=ann.bbox_h)

    polygon = json.loads(ann.polygon_points) if ann.polygon_points else None
    mask = json.loads(ann.mask_rle) if ann.mask_rle else None

    return AnnotationResponse(
        id=ann.id,
        image_id=ann.image_id,
        class_name=ann.class_name,
        annotation_type=ann.annotation_type,
        bbox=bbox,
        polygon_points=polygon,
        mask_rle=mask,
        confidence=ann.confidence,
        source=ann.source,
        is_verified=ann.is_verified,
        created_at=ann.created_at,
    )


@router.post("/", response_model=AnnotationResponse, status_code=201)
async def create_annotation(data: AnnotationCreate, db: DbSession):
    ann = Annotation(
        image_id=data.image_id,
        class_name=data.class_name,
        annotation_type=data.annotation_type,
        confidence=data.confidence,
        source=data.source,
    )

    if data.bbox:
        ann.bbox_x = data.bbox.x
        ann.bbox_y = data.bbox.y
        ann.bbox_w = data.bbox.w
        ann.bbox_h = data.bbox.h

    if data.polygon_points:
        ann.polygon_points = json.dumps(data.polygon_points)

    if data.mask_rle:
        ann.mask_rle = json.dumps(data.mask_rle)

    db.add(ann)
    await db.flush()
    await db.refresh(ann)

    # Update image status
    result = await db.execute(select(Image).where(Image.id == data.image_id))
    image = result.scalar_one_or_none()
    if image and image.status == "pending":
        image.status = "annotated"

    return _annotation_to_response(ann)


@router.get("/image/{image_id}", response_model=list[AnnotationResponse])
async def get_image_annotations(image_id: uuid.UUID, db: DbSession):
    result = await db.execute(select(Annotation).where(Annotation.image_id == image_id))
    annotations = result.scalars().all()
    return [_annotation_to_response(a) for a in annotations]


@router.patch("/{annotation_id}", response_model=AnnotationResponse)
async def update_annotation(annotation_id: uuid.UUID, data: AnnotationUpdate, db: DbSession):
    result = await db.execute(select(Annotation).where(Annotation.id == annotation_id))
    ann = result.scalar_one_or_none()
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")

    if data.class_name is not None:
        ann.class_name = data.class_name
    if data.bbox is not None:
        ann.bbox_x = data.bbox.x
        ann.bbox_y = data.bbox.y
        ann.bbox_w = data.bbox.w
        ann.bbox_h = data.bbox.h
    if data.is_verified is not None:
        ann.is_verified = data.is_verified

    await db.flush()
    await db.refresh(ann)
    return _annotation_to_response(ann)


@router.delete("/{annotation_id}", status_code=204)
async def delete_annotation(annotation_id: uuid.UUID, db: DbSession):
    result = await db.execute(select(Annotation).where(Annotation.id == annotation_id))
    ann = result.scalar_one_or_none()
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")
    await db.delete(ann)


@router.post("/batch-delete", status_code=204)
async def batch_delete_annotations(image_id: uuid.UUID, db: DbSession):
    """Delete all annotations for an image."""
    result = await db.execute(select(Annotation).where(Annotation.image_id == image_id))
    annotations = result.scalars().all()
    for ann in annotations:
        await db.delete(ann)


@router.post("/batch-verify")
async def batch_verify_annotations(annotation_ids: list[uuid.UUID], db: DbSession):
    """Mark multiple annotations as verified."""
    verified = 0
    for ann_id in annotation_ids:
        result = await db.execute(select(Annotation).where(Annotation.id == ann_id))
        ann = result.scalar_one_or_none()
        if ann:
            ann.is_verified = True
            verified += 1
    return {"verified": verified}


@router.get("/export/{project_id}")
async def export_annotations(
    project_id: uuid.UUID,
    db: DbSession,
    format: str = "yolo",
):
    """Export annotations in various formats (yolo, coco, voc, csv)."""
    from app.models.image import Image

    result = await db.execute(select(Image).where(Image.project_id == project_id))
    images = result.scalars().all()

    if format == "yolo":
        export_data = []
        for img in images:
            ann_result = await db.execute(select(Annotation).where(Annotation.image_id == img.id))
            anns = ann_result.scalars().all()
            image_annotations = []
            for ann in anns:
                if ann.bbox_x is not None:
                    image_annotations.append({
                        "class": ann.class_name,
                        "x_center": ann.bbox_x,
                        "y_center": ann.bbox_y,
                        "width": ann.bbox_w,
                        "height": ann.bbox_h,
                    })
            export_data.append({
                "image": img.filename,
                "annotations": image_annotations,
            })
        return {"format": "yolo", "data": export_data}

    elif format == "coco":
        coco = {
            "images": [],
            "annotations": [],
            "categories": [],
        }
        category_map = {}
        ann_id_counter = 1

        for img in images:
            coco["images"].append({
                "id": str(img.id),
                "file_name": img.filename,
                "width": img.width,
                "height": img.height,
            })
            ann_result = await db.execute(select(Annotation).where(Annotation.image_id == img.id))
            for ann in ann_result.scalars().all():
                if ann.class_name not in category_map:
                    cat_id = len(category_map) + 1
                    category_map[ann.class_name] = cat_id
                    coco["categories"].append({"id": cat_id, "name": ann.class_name})

                if ann.bbox_x is not None:
                    x = (ann.bbox_x - ann.bbox_w / 2) * img.width
                    y = (ann.bbox_y - ann.bbox_h / 2) * img.height
                    w = ann.bbox_w * img.width
                    h = ann.bbox_h * img.height
                    coco["annotations"].append({
                        "id": ann_id_counter,
                        "image_id": str(img.id),
                        "category_id": category_map[ann.class_name],
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    })
                    ann_id_counter += 1

        return {"format": "coco", "data": coco}

    raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
