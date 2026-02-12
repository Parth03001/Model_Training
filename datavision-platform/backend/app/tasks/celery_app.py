"""Celery application configuration.

When celery is not installed (local dev without Redis), provides a stub
that lets @celery_app.task() decorate functions as plain callables.
"""

import uuid
import threading

from app.config import settings

try:
    from celery import Celery
    from celery.signals import worker_init

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

    @worker_init.connect
    def warmup_models(**kwargs):
        """Pre-load GPU models when the Celery worker starts.
        This avoids the 3-30s model loading delay on the first task."""
        from loguru import logger
        logger.info("Worker starting — warming up GPU models...")
        try:
            from app.services.auto_annotate.sam2 import _load_model as load_sam2
            load_sam2()
        except Exception as e:
            logger.warning(f"SAM2 warmup skipped: {e}")
        try:
            from app.services.auto_annotate.clip_search import _load_model as load_clip
            load_clip()
        except Exception as e:
            logger.warning(f"CLIP warmup skipped: {e}")
        try:
            from app.services.auto_annotate.grounding_dino import _load_model as load_gdino
            load_gdino()
        except Exception as e:
            logger.warning(f"Grounding DINO warmup skipped: {e}")
        logger.info("Model warmup complete")

    CELERY_AVAILABLE = True

except ImportError:
    # Stub for local dev without Celery/Redis.
    # Tasks run in background threads so the API doesn't block.

    class _StubResult:
        """Mimics a Celery AsyncResult for async-like execution."""
        def __init__(self, task_id: str | None = None):
            self.id = task_id or str(uuid.uuid4())
            self.result = None
            self.status = "PENDING"
            self._ready = threading.Event()

        def ready(self):
            return self._ready.is_set()

        def successful(self):
            return self.status == "SUCCESS"

        def _set_result(self, result):
            self.result = result
            self.status = "SUCCESS"
            self._ready.set()

        def _set_failure(self, error):
            self.result = str(error)
            self.status = "FAILURE"
            self._ready.set()

    # Track results for task status polling
    _task_results: dict[str, _StubResult] = {}

    class _StubControl:
        def revoke(self, *args, **kwargs):
            pass

    class _CeleryStub:
        control = _StubControl()

        def task(self, *args, **kwargs):
            def decorator(fn):
                def _delay(*a, **kw):
                    stub_result = _StubResult()
                    _task_results[stub_result.id] = stub_result

                    def _run():
                        try:
                            result = fn(None, *a, **kw)
                            stub_result._set_result(result)
                        except Exception as e:
                            stub_result._set_failure(e)

                    thread = threading.Thread(target=_run, daemon=True)
                    thread.start()
                    return stub_result

                def _apply_async(args=(), kwargs=None, **_):
                    stub_result = _StubResult()
                    _task_results[stub_result.id] = stub_result

                    def _run():
                        try:
                            result = fn(None, *args, **(kwargs or {}))
                            stub_result._set_result(result)
                        except Exception as e:
                            stub_result._set_failure(e)

                    thread = threading.Thread(target=_run, daemon=True)
                    thread.start()
                    return stub_result

                fn.delay = _delay
                fn.apply_async = _apply_async
                return fn
            return decorator

        def autodiscover_tasks(self, *args, **kwargs):
            pass

        def AsyncResult(self, task_id):
            if task_id in _task_results:
                return _task_results[task_id]
            return _StubResult(task_id)

    celery_app = _CeleryStub()

    CELERY_AVAILABLE = False
