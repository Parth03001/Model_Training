"""Annotation Pydantic schemas."""

import uuid
from datetime import datetime
from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Normalized bounding box [0-1]."""
    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)
    w: float = Field(..., ge=0, le=1)
    h: float = Field(..., ge=0, le=1)


class AnnotationCreate(BaseModel):
    image_id: uuid.UUID
    class_name: str
    annotation_type: str = Field(..., pattern="^(bbox|polygon|mask|classification)$")
    bbox: BBox | None = None
    polygon_points: list[list[float]] | None = None
    mask_rle: dict | None = None
    confidence: float | None = None
    source: str = "manual"


class AnnotationUpdate(BaseModel):
    class_name: str | None = None
    bbox: BBox | None = None
    polygon_points: list[list[float]] | None = None
    is_verified: bool | None = None


class AnnotationResponse(BaseModel):
    id: uuid.UUID
    image_id: uuid.UUID
    class_name: str
    annotation_type: str
    bbox: BBox | None = None
    polygon_points: list[list[float]] | None = None
    mask_rle: dict | None = None
    confidence: float | None
    source: str
    is_verified: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AutoAnnotateRequest(BaseModel):
    """Request for auto-annotation using Grounding DINO."""
    image_ids: list[uuid.UUID]
    text_prompt: str = Field(..., min_length=1, description="Classes separated by periods, e.g. 'hard hat . person .'")
    box_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    text_threshold: float = Field(default=0.25, ge=0.0, le=1.0)


class SAMPromptRequest(BaseModel):
    """Request for SAM2 mask generation from box/point prompts."""
    image_id: uuid.UUID
    box_prompts: list[BBox] | None = None
    point_prompts: list[dict] | None = None  # [{"x": 0.5, "y": 0.5, "label": 1}]


class CLIPSearchRequest(BaseModel):
    """Request for CLIP-based visual similarity search."""
    image_id: uuid.UUID
    crop_bbox: BBox
    class_name: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class GroundedSAMRequest(BaseModel):
    """Combined Grounding DINO + SAM2 pipeline."""
    image_ids: list[uuid.UUID]
    text_prompt: str
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    generate_masks: bool = True


class AutoAnnotateResponse(BaseModel):
    task_id: str
    status: str = "queued"
    message: str


class TrainedModelInferenceRequest(BaseModel):
    """Request for auto-annotation using a project's latest trained model."""
    project_id: uuid.UUID
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    image_ids: list[uuid.UUID] | None = None  # None means all images without annotations
