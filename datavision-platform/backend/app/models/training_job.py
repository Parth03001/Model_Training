"""Training job database model."""

import uuid
from datetime import datetime

from sqlalchemy import String, Integer, Float, Text, DateTime, ForeignKey, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Model configuration
    model_architecture: Mapped[str] = mapped_column(
        SAEnum("yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
               "yolov11n", "yolov11s", "yolov11m", "yolov11l",
               "vit_b_16", "vit_b_32", "vit_l_16",
               "efficientnet_v2_s", "efficientnet_v2_m",
               "rt_detr_l", "rt_detr_x",
               name="model_arch_enum"),
        nullable=False,
    )
    task_type: Mapped[str] = mapped_column(
        SAEnum("classification", "detection", "segmentation", name="training_task_enum"),
        nullable=False,
    )

    # Hyperparameters
    epochs: Mapped[int] = mapped_column(Integer, default=100)
    batch_size: Mapped[int] = mapped_column(Integer, default=16)
    img_size: Mapped[int] = mapped_column(Integer, default=640)
    learning_rate: Mapped[float] = mapped_column(Float, default=0.01)
    patience: Mapped[int] = mapped_column(Integer, default=20)
    optimizer: Mapped[str] = mapped_column(String(50), default="AdamW")
    augmentation_config: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON

    # Status & progress
    status: Mapped[str] = mapped_column(
        SAEnum("queued", "preparing", "training", "completed", "failed", "cancelled", name="job_status_enum"),
        default="queued",
    )
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
