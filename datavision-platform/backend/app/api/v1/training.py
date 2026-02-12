"""Training job API routes."""

import json
import uuid

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.api.deps import DbSession
from app.models.training_job import TrainingJob
from app.schemas.training import TrainingJobCreate, TrainingJobResponse, ExportRequest
from app.tasks.training_tasks import run_training_job, run_model_export

router = APIRouter()


def _job_to_response(job: TrainingJob) -> TrainingJobResponse:
    return TrainingJobResponse(
        id=job.id,
        project_id=job.project_id,
        name=job.name,
        model_architecture=job.model_architecture,
        task_type=job.task_type,
        epochs=job.epochs,
        batch_size=job.batch_size,
        img_size=job.img_size,
        learning_rate=job.learning_rate,
        patience=job.patience,
        optimizer=job.optimizer,
        status=job.status,
        current_epoch=job.current_epoch,
        metrics=json.loads(job.metrics) if job.metrics else None,
        best_metrics=json.loads(job.best_metrics) if job.best_metrics else None,
        model_path=job.model_path,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.post("/jobs", response_model=TrainingJobResponse, status_code=201)
async def create_training_job(data: TrainingJobCreate, db: DbSession):
    """Create and queue a new training job."""
    job = TrainingJob(
        project_id=data.project_id,
        name=data.name,
        model_architecture=data.model_architecture,
        task_type=data.task_type,
        epochs=data.epochs,
        batch_size=data.batch_size,
        img_size=data.img_size,
        learning_rate=data.learning_rate,
        patience=data.patience,
        optimizer=data.optimizer,
        augmentation_config=json.dumps(data.augmentation.model_dump()) if data.augmentation else None,
        status="queued",
    )
    db.add(job)
    await db.flush()
    await db.refresh(job)

    # Queue Celery task
    task = run_training_job.delay(str(job.id))
    job.celery_task_id = task.id
    await db.flush()

    return _job_to_response(job)


@router.get("/jobs", response_model=list[TrainingJobResponse])
async def list_training_jobs(
    db: DbSession,
    project_id: uuid.UUID | None = None,
    status: str | None = None,
):
    query = select(TrainingJob)
    if project_id:
        query = query.where(TrainingJob.project_id == project_id)
    if status:
        query = query.where(TrainingJob.status == status)
    query = query.order_by(TrainingJob.created_at.desc())

    result = await db.execute(query)
    jobs = result.scalars().all()
    return [_job_to_response(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: uuid.UUID, db: DbSession):
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return _job_to_response(job)


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: uuid.UUID, db: DbSession):
    """Cancel a running or queued training job."""
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job.status not in ("queued", "training", "preparing"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status: {job.status}")

    # Revoke Celery task
    if job.celery_task_id:
        from app.tasks.celery_app import celery_app
        celery_app.control.revoke(job.celery_task_id, terminate=True)

    job.status = "cancelled"
    await db.flush()
    return {"status": "cancelled"}


@router.post("/export")
async def export_model(request: ExportRequest):
    """Export a trained model to various formats."""
    task = run_model_export.delay(
        model_id=str(request.model_id),
        formats=request.formats,
        img_size=request.img_size,
        quantize=request.quantize,
    )
    return {"task_id": task.id, "status": "queued", "message": "Model export queued"}


@router.websocket("/ws/training/{job_id}")
async def training_metrics_ws(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time training metrics.
    Frontend connects here to receive live epoch updates.
    Requires Redis; falls back to polling DB when Redis is unavailable.
    """
    await websocket.accept()
    import asyncio

    try:
        import redis.asyncio as aioredis
    except ImportError:
        # No redis — fall back to periodic DB polling
        from app.api.deps import async_session_factory
        try:
            while True:
                async with async_session_factory() as db:
                    result = await db.execute(
                        select(TrainingJob).where(TrainingJob.id == job_id)
                    )
                    job = result.scalar_one_or_none()
                    if job:
                        import json as _json
                        payload = {
                            "status": job.status,
                            "current_epoch": job.current_epoch,
                            "metrics": _json.loads(job.metrics) if job.metrics else None,
                        }
                        await websocket.send_json(payload)
                        if job.status in ("completed", "failed", "cancelled"):
                            break
                await asyncio.sleep(2)
        except WebSocketDisconnect:
            pass
        return

    from app.config import settings as _settings

    r = aioredis.from_url(_settings.redis_url)
    channel = f"training:{job_id}:metrics"

    pubsub = r.pubsub()
    await pubsub.subscribe(channel)

    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message["type"] == "message":
                await websocket.send_text(message["data"].decode())
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(channel)
        await r.close()
