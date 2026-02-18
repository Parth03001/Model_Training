"""Image upload and management API routes."""

import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from PIL import Image as PILImage
from sqlalchemy import select, func

from app.api.deps import DbSession
from app.config import settings
from app.models.image import Image
from app.models.annotation import Annotation
from app.schemas.image import ImageResponse, ImageListResponse, ImageUploadResponse

router = APIRouter(redirect_slashes=False)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _image_to_response(image: Image, annotation_count: int = 0) -> ImageResponse:
    return ImageResponse(
        id=image.id,
        project_id=image.project_id,
        filename=image.filename,
        filepath=f"/uploads/{image.project_id}/{image.filename}",
        width=image.width,
        height=image.height,
        file_size=image.file_size,
        status=image.status,
        split=image.split,
        annotation_count=annotation_count,
        created_at=image.created_at,
    )


@router.post("/upload", response_model=ImageUploadResponse)
async def upload_images(
    db: DbSession,
    project_id: uuid.UUID = Form(...),
    files: list[UploadFile] = File(...),
):
    """Upload one or more images to a project."""
    project_dir = settings.upload_dir / str(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)

    uploaded_images = []
    failed = 0

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            failed += 1
            continue

        file_path = project_dir / file.filename
        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # Get image dimensions
            with PILImage.open(file_path) as img:
                width, height = img.size

            file_size = file_path.stat().st_size

            db_image = Image(
                project_id=project_id,
                filename=file.filename,
                filepath=str(file_path),
                width=width,
                height=height,
                file_size=file_size,
                status="pending",
            )
            db.add(db_image)
            await db.flush()
            await db.refresh(db_image)
            uploaded_images.append(_image_to_response(db_image))
        except Exception:
            failed += 1
            if file_path.exists():
                file_path.unlink()

    return ImageUploadResponse(
        uploaded=len(uploaded_images),
        failed=failed,
        images=uploaded_images,
    )


@router.get("/project/{project_id}", response_model=ImageListResponse)
async def list_project_images(
    project_id: uuid.UUID,
    db: DbSession,
    skip: int = 0,
    limit: int = 50,
    status: str | None = None,
    split: str | None = None,
):
    query = select(Image).where(Image.project_id == project_id)
    if status:
        query = query.where(Image.status == status)
    if split:
        query = query.where(Image.split == split)
    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    images = result.scalars().all()

    count_q = select(func.count(Image.id)).where(Image.project_id == project_id)
    count_result = await db.execute(count_q)
    total = count_result.scalar() or 0

    items = []
    for img in images:
        ann_count_q = await db.execute(
            select(func.count(Annotation.id)).where(Annotation.image_id == img.id)
        )
        ann_count = ann_count_q.scalar() or 0
        items.append(_image_to_response(img, ann_count))

    return ImageListResponse(images=items, total=total)


@router.get("/{image_id}", response_model=ImageResponse)
async def get_image(image_id: uuid.UUID, db: DbSession):
    result = await db.execute(select(Image).where(Image.id == image_id))
    image = result.scalar_one_or_none()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    ann_count_q = await db.execute(
        select(func.count(Annotation.id)).where(Annotation.image_id == image_id)
    )
    ann_count = ann_count_q.scalar() or 0
    return _image_to_response(image, ann_count)


@router.delete("/{image_id}", status_code=204)
async def delete_image(image_id: uuid.UUID, db: DbSession):
    result = await db.execute(select(Image).where(Image.id == image_id))
    image = result.scalar_one_or_none()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Delete file from disk
    file_path = Path(image.filepath)
    if file_path.exists():
        file_path.unlink()

    await db.delete(image)


@router.patch("/{image_id}/split")
async def set_image_split(image_id: uuid.UUID, split: str, db: DbSession):
    """Assign an image to train/val/test split."""
    if split not in ("train", "val", "test"):
        raise HTTPException(status_code=400, detail="Split must be train, val, or test")

    result = await db.execute(select(Image).where(Image.id == image_id))
    image = result.scalar_one_or_none()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    image.split = split
    await db.flush()
    return {"status": "ok", "split": split}
