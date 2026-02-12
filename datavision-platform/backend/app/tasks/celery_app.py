"""Celery application configuration."""

from celery import Celery

from app.config import settings

celery_app = Celery(
    "datavision",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Worker settings (GPU-friendly)
    worker_prefetch_multiplier=1,       # Fetch one task at a time
    worker_concurrency=1,                # One GPU task at a time
    task_acks_late=True,                 # Acknowledge after completion
    task_reject_on_worker_lost=True,     # Re-queue if worker dies

    # Task routing
    task_routes={
        "app.tasks.auto_annotate_tasks.*": {"queue": "auto_annotate"},
        "app.tasks.training_tasks.run_training_job": {"queue": "training"},
        "app.tasks.training_tasks.run_model_export": {"queue": "export"},
        "app.tasks.export_tasks.*": {"queue": "export"},
    },

    # Result expiration
    result_expires=86400,  # 24 hours

    # Timezone
    timezone="UTC",
    enable_utc=True,
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["app.tasks"])
