"""Celery application configuration.

Priority order:
  1. Real Celery + live worker  → tasks processed by the worker process
  2. Celery installed, no worker → fall back to in-process daemon threads
  3. Celery not installed        → fall back to in-process daemon threads

The in-process stub stores results in a module-level dict so that the
/auto-annotate/task/{task_id} endpoint can still poll for completion.
"""

import uuid
import threading
from loguru import logger

from app.config import settings

# ---------------------------------------------------------------------------
# In-process stub — used as fallback when no Celery worker is reachable
# ---------------------------------------------------------------------------

class _StubResult:
    """Mimics a Celery AsyncResult for thread-based execution."""
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

    def update_state(self, state=None, meta=None):
        """Mimics Celery's Task.update_state."""
        if state:
            self.status = state
        if meta:
            self.result = meta


# Module-level store so AsyncResult lookups work across requests
_task_results: dict[str, _StubResult] = {}


class _StubControl:
    def revoke(self, *args, **kwargs):
        pass


class _CeleryStub:
    """Drop-in replacement for a Celery app when no worker is available."""

    control = _StubControl()

    def task(self, *args, **kwargs):
        def decorator(fn):
            def _delay(*a, **kw):
                stub = _StubResult()
                _task_results[stub.id] = stub

                def _run():
                    try:
                        stub.status = "STARTED"
                        # Pass stub as 'self' if the task is bound
                        result = fn(stub, *a, **kw)
                        stub._set_result(result)
                    except Exception as e:
                        logger.exception(f"In-process task {fn.__name__} failed: {e}")
                        stub._set_failure(e)

                threading.Thread(target=_run, daemon=True).start()
                return stub

            def _apply_async(args=(), kwargs=None, **_):
                stub = _StubResult()
                _task_results[stub.id] = stub

                def _run():
                    try:
                        stub.status = "STARTED"
                        # Pass stub as 'self'
                        result = fn(stub, *args, **(kwargs or {}))
                        stub._set_result(result)
                    except Exception as e:
                        logger.exception(f"In-process task {fn.__name__} failed: {e}")
                        stub._set_failure(e)

                threading.Thread(target=_run, daemon=True).start()
                return stub

            fn.delay = _delay
            fn.apply_async = _apply_async
            return fn

        return decorator

    def autodiscover_tasks(self, *args, **kwargs):
        pass

    def AsyncResult(self, task_id):
        # Return the live stub if we have it; otherwise return a dead stub
        # so the status endpoint returns PENDING rather than crashing.
        return _task_results.get(task_id, _StubResult(task_id))


# ---------------------------------------------------------------------------
# Try to create a real Celery app backed by a live worker
# ---------------------------------------------------------------------------

def _build_real_celery():
    """
    Create and validate a real Celery application.
    Raises RuntimeError if no workers respond within 1 second.
    """
    from celery import Celery
    from celery.signals import worker_init

    app = Celery(
        "datavision",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
    )

    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        worker_prefetch_multiplier=1,
        worker_concurrency=1,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_routes={
            "app.tasks.auto_annotate_tasks.*": {"queue": "auto_annotate"},
            "app.tasks.training_tasks.run_training_job": {"queue": "training"},
            "app.tasks.training_tasks.run_model_export": {"queue": "export"},
            "app.tasks.export_tasks.*": {"queue": "export"},
        },
        result_expires=86400,
        timezone="UTC",
        enable_utc=True,
    )

    app.autodiscover_tasks(["app.tasks"])

    @worker_init.connect
    def warmup_models(**kwargs):
        """Pre-load GPU models when the Celery worker starts."""
        logger.info("Worker starting — warming up GPU models...")
        for name, loader_path in [
            ("SAM2", "app.services.auto_annotate.sam2._load_model"),
            ("CLIP", "app.services.auto_annotate.clip_search._load_model"),
            ("Grounding DINO", "app.services.auto_annotate.grounding_dino._load_model"),
        ]:
            try:
                module_path, fn_name = loader_path.rsplit(".", 1)
                import importlib
                mod = importlib.import_module(module_path)
                getattr(mod, fn_name)()
            except Exception as e:
                logger.warning(f"{name} warmup skipped: {e}")
        logger.info("Model warmup complete")

    # Ping for live workers — returns {} if none respond
    try:
        ping = app.control.ping(timeout=1.0)
    except Exception as e:
        raise RuntimeError(f"Celery broker unreachable: {e}") from e

    if not ping:
        raise RuntimeError("Celery installed but no workers are running")

    return app


CELERY_AVAILABLE = False

try:
    celery_app = _build_real_celery()
    CELERY_AVAILABLE = True
    logger.info("Celery worker detected — tasks will be processed by the worker")
except ImportError:
    logger.info("Celery not installed — using in-process threading for background tasks")
    celery_app = _CeleryStub()
except RuntimeError as e:
    logger.warning(f"{e} — using in-process threading for background tasks")
    celery_app = _CeleryStub()
