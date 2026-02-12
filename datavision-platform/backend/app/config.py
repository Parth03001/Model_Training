"""Application configuration loaded from environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Database ---
    database_url: str = "postgresql+asyncpg://datavision:changeme@db:5432/datavision"

    # --- Redis ---
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"

    # --- Storage ---
    upload_dir: Path = Path("/data/uploads")
    model_dir: Path = Path("/data/models")
    export_dir: Path = Path("/data/exports")
    faiss_index_dir: Path = Path("/data/faiss_indices")

    # --- Grounding DINO ---
    grounding_dino_model: str = "IDEA-Research/grounding-dino-base"
    grounding_dino_box_threshold: float = 0.3
    grounding_dino_text_threshold: float = 0.25

    # --- SAM 2 ---
    sam2_model: str = "facebook/sam2-hiera-large"
    sam2_checkpoint: str = "sam2_hiera_large.pt"

    # --- CLIP ---
    clip_model: str = "openai/clip-vit-large-patch14"

    # --- Training defaults ---
    default_epochs: int = 100
    default_batch_size: int = 16
    default_img_size: int = 640
    default_patience: int = 20

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
