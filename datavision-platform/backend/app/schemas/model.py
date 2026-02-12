"""Trained model Pydantic schemas."""

import uuid
from datetime import datetime
from pydantic import BaseModel


class TrainedModelResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    training_job_id: uuid.UUID | None
    name: str
    architecture: str
    task_type: str
    version: str
    accuracy: float | None
    map50: float | None
    map50_95: float | None
    precision_val: float | None
    recall_val: float | None
    f1_score: float | None
    inference_time_ms: float | None
    model_path: str
    model_size_mb: float | None
    export_format: str
    num_classes: int
    class_names: list[str] | None = None
    description: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelListResponse(BaseModel):
    models: list[TrainedModelResponse]
    total: int


class ModelCompareRequest(BaseModel):
    model_ids: list[uuid.UUID]


class ModelCompareResponse(BaseModel):
    models: list[TrainedModelResponse]
    comparison_metrics: dict
