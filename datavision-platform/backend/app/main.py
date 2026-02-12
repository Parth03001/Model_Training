"""DataVision Platform — FastAPI application entry point."""

import sys
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import settings
from app.database import init_db
from app.api.v1.router import api_router


# ---------------------------------------------------------------------------
# Logging setup — route everything through loguru
# ---------------------------------------------------------------------------

def setup_logging():
    """Configure loguru as the single logging sink.

    - Removes loguru's default handler (avoids duplicate output)
    - Adds a new stderr handler with color, timestamp, and configurable level
    - Intercepts Python's stdlib `logging` so uvicorn/sqlalchemy/celery logs
      also flow through loguru's formatter
    """

    # Remove default loguru handler
    logger.remove()

    # Add a single formatted handler to stderr
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Intercept stdlib logging → loguru
    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            # Map stdlib log level to loguru level
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            # Find caller from where the log was issued
            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Redirect specific noisy loggers
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "sqlalchemy.engine"):
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False


setup_logging()


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Skip noisy static file / health requests at DEBUG level
    path = request.url.path
    if path.startswith("/uploads") or path == "/health":
        logger.debug("{method} {path} → {status} ({ms:.0f}ms)",
                     method=request.method, path=path,
                     status=response.status_code, ms=elapsed_ms)
    else:
        logger.info("{method} {path} → {status} ({ms:.0f}ms)",
                    method=request.method, path=path,
                    status=response.status_code, ms=elapsed_ms)

    return response


# --- Static file serving for uploaded images ---
app.mount("/uploads", StaticFiles(directory=str(settings.upload_dir)), name="uploads")

# --- API routes ---
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}
