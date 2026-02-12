"""Trained model registry database model."""

import uuid
from datetime import datetime

from sqlalchemy import String, Float, Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.base import UUIDType


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id: Mapped[uuid.UUID] = mapped_column(UUIDType, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(UUIDType, ForeignKey("projects.id"), nullable=False)
    training_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUIDType, ForeignKey("training_jobs.id"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    architecture: Mapped[str] = mapped_column(String(100), nullable=False)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[str] = mapped_column(String(50), default="1.0.0")

    # Metrics
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    map50: Mapped[float | None] = mapped_column(Float, nullable=True)
    map50_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_val: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_val: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # File info
    model_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    model_size_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    export_format: Mapped[str] = mapped_column(String(50), default="pytorch")  # pytorch, onnx, tflite, torchscript
    num_classes: Mapped[int] = mapped_column(Integer, nullable=False)
    class_names: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON

    # Metadata
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
