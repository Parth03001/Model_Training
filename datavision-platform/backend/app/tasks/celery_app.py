"""Celery application configuration.

When celery is not installed (local dev without Redis), provides a stub
that lets @celery_app.task() decorate functions as plain callables.
"""

import uuid

from app.config import settings

try:
    from celery import Celery

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

    CELERY_AVAILABLE = True

except ImportError:
    # Stub for local dev without Celery/Redis.
    # Tasks run synchronously; .delay() returns a simple result object.

    class _StubResult:
        """Mimics a Celery AsyncResult for synchronous execution."""
        def __init__(self, result):
            self.id = str(uuid.uuid4())
            self.result = result
            self.status = "SUCCESS"

        def ready(self):
            return True

        def successful(self):
            return True

    class _StubControl:
        def revoke(self, *args, **kwargs):
            pass

    class _CeleryStub:
        control = _StubControl()

        def task(self, *args, **kwargs):
            def decorator(fn):
                def _delay(*a, **kw):
                    result = fn(None, *a, **kw)
                    return _StubResult(result)

                def _apply_async(args=(), kwargs=None, **_):
                    result = fn(None, *args, **(kwargs or {}))
                    return _StubResult(result)

                fn.delay = _delay
                fn.apply_async = _apply_async
                return fn
            return decorator

        def autodiscover_tasks(self, *args, **kwargs):
            pass

        def AsyncResult(self, task_id):
            return _StubResult({"status": "local_dev", "message": "Celery not available"})

    celery_app = _CeleryStub()

    CELERY_AVAILABLE = False
