"""DataVision Platform — FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import settings
from app.database import init_db
from app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Starting DataVision Platform...")

    # Ensure storage directories exist
    for d in [settings.upload_dir, settings.model_dir, settings.export_dir, settings.faiss_index_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create DB tables (dev mode — use Alembic migrations in production)
    await init_db()
    logger.info("Database tables created.")

    yield

    logger.info("Shutting down DataVision Platform.")


app = FastAPI(
    title="DataVision Platform",
    description="Auto-annotation and model training platform for computer vision",
    version="0.1.0",
    lifespan=lifespan,
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static file serving for uploaded images ---
app.mount("/uploads", StaticFiles(directory=str(settings.upload_dir)), name="uploads")

# --- API routes ---
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}
