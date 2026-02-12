"""Training job database model."""

import uuid
from datetime import datetime

from sqlalchemy import String, Integer, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import UUIDType


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUIDType, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(UUIDType, ForeignKey("projects.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Model configuration
    model_architecture: Mapped[str] = mapped_column(String(100), nullable=False)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Hyperparameters
    epochs: Mapped[int] = mapped_column(Integer, default=100)
    batch_size: Mapped[int] = mapped_column(Integer, default=16)
    img_size: Mapped[int] = mapped_column(Integer, default=640)
    learning_rate: Mapped[float] = mapped_column(Float, default=0.01)
    patience: Mapped[int] = mapped_column(Integer, default=20)
    optimizer: Mapped[str] = mapped_column(String(50), default="AdamW")
    augmentation_config: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON

    # Status & progress
    status: Mapped[str] = mapped_column(String(50), default="queued")
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Metrics (JSON)
    metrics: Mapped[str | None] = mapped_column(Text, nullable=True)
    best_metrics: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Output
    model_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    export_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    project = relationship("Project", back_populates="training_jobs")
