"""File storage utilities."""

from pathlib import Path
from app.config import settings


def get_project_upload_dir(project_id: str) -> Path:
    """Get the upload directory for a project."""
    path = settings.upload_dir / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_dir(job_id: str) -> Path:
    """Get the model output directory for a training job."""
    path = settings.model_dir / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_export_dir(model_id: str) -> Path:
    """Get the export directory for a model."""
    path = settings.export_dir / model_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_project_files(project_id: str):
    """Remove all files associated with a project."""
    import shutil

    upload_dir = settings.upload_dir / project_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)

    model_dir = settings.model_dir / project_id
    if model_dir.exists():
        shutil.rmtree(model_dir)

    faiss_dir = settings.faiss_index_dir / project_id
    if faiss_dir.exists():
        shutil.rmtree(faiss_dir)
