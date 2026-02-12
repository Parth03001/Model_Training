"""Annotation database model."""

import uuid
from datetime import datetime

from sqlalchemy import String, Float, Text, DateTime, ForeignKey, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=False)
    class_name: Mapped[str] = mapped_column(String(255), nullable=False)
    annotation_type: Mapped[str] = mapped_column(
        SAEnum("bbox", "polygon", "mask", "classification", name="annotation_type_enum"),
        nullable=False,
    )

    # Bounding box (normalized 0-1): x_center, y_center, width, height
    bbox_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_w: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_h: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Polygon points as JSON: [[x1,y1],[x2,y2],...]
    polygon_points: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Segmentation mask as RLE-encoded JSON
    mask_rle: Mapped[str | None] = mapped_column(Text, nullable=True)

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(
        SAEnum("manual", "grounding_dino", "sam2", "grounded_sam", "clip", "model", name="annotation_source_enum"),
        default="manual",
    )
    is_verified: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    image = relationship("Image", back_populates="annotations")
