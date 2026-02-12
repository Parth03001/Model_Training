"""API v1 router — aggregates all sub-routers."""

from fastapi import APIRouter

from app.api.v1 import projects, images, annotations, auto_annotate, training, models

api_router = APIRouter()

api_router.include_router(projects.router, prefix="/projects", tags=["Projects"])
api_router.include_router(images.router, prefix="/images", tags=["Images"])
api_router.include_router(annotations.router, prefix="/annotations", tags=["Annotations"])
api_router.include_router(auto_annotate.router, prefix="/auto-annotate", tags=["Auto-Annotation"])
api_router.include_router(training.router, prefix="/training", tags=["Training"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])
