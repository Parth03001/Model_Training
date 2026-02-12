"""Training Pydantic schemas."""

import uuid
from datetime import datetime
from pydantic import BaseModel, Field


class AugmentationConfig(BaseModel):
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0


class TrainingJobCreate(BaseModel):
    project_id: uuid.UUID
    name: str = Field(..., min_length=1, max_length=255)
    model_architecture: str
    task_type: str = Field(..., pattern="^(classification|detection|segmentation)$")
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1, le=128)
    img_size: int = Field(default=640, ge=32, le=1280)
    learning_rate: float = Field(default=0.01, gt=0, le=1)
    patience: int = Field(default=20, ge=1, le=100)
    optimizer: str = Field(default="AdamW", pattern="^(SGD|Adam|AdamW|RMSprop)$")
    augmentation: AugmentationConfig | None = None
    train_split: float = Field(default=0.8, ge=0.5, le=0.95)
    val_split: float = Field(default=0.15, ge=0.05, le=0.4)
    test_split: float = Field(default=0.05, ge=0.0, le=0.2)


class TrainingJobResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    model_architecture: str
    task_type: str
    epochs: int
    batch_size: int
    img_size: int
    learning_rate: float
    patience: int
    optimizer: str
    status: str
    current_epoch: int
    metrics: dict | None = None
    best_metrics: dict | None = None
    model_path: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    model_config = {"from_attributes": True}


class TrainingMetricsUpdate(BaseModel):
    """Real-time metrics sent via WebSocket during training."""
    epoch: int
    train_loss: float
    val_loss: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    precision: float | None = None
    recall: float | None = None
    accuracy: float | None = None
    learning_rate: float | None = None


class ExportRequest(BaseModel):
    model_id: uuid.UUID
    formats: list[str] = Field(
        default=["onnx"],
        description="Export formats: onnx, tflite, torchscript, coreml"
    )
    img_size: int = 640
    quantize: bool = False
