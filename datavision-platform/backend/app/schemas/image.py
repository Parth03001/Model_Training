"""Image Pydantic schemas."""

import uuid
from datetime import datetime
from pydantic import BaseModel


class ImageResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    filename: str
    filepath: str
    width: int
    height: int
    file_size: int
    status: str
    split: str | None
    annotation_count: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


class ImageListResponse(BaseModel):
    images: list[ImageResponse]
    total: int


class ImageUploadResponse(BaseModel):
    uploaded: int
    failed: int
    images: list[ImageResponse]
