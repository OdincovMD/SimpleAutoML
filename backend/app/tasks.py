import json
import os
import urllib.error
import urllib.request

from celery import Celery
from backend.config import settings

celery_app = Celery(
    "automl",
    broker=settings.celery_broker,
    backend=settings.REDIS_URL,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)


def _notify_backend_storage_sync(job_id: str, task_folder: str) -> None:
    """Worker не ходит в MinIO — только backend (микросервис хранилища по сети)."""
    backend_url = os.environ.get("BACKEND_INTERNAL_URL", "http://backend:8000").rstrip("/")
    token = (os.environ.get("INTERNAL_STORAGE_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("INTERNAL_STORAGE_TOKEN is not set")
    payload = json.dumps({"job_id": job_id, "task_folder": task_folder}).encode()
    req = urllib.request.Request(
        f"{backend_url}/api/internal/storage/sync",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Internal-Token": token,
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=3600)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"Storage sync failed: HTTP {e.code} {body}") from e


@celery_app.task(bind=True)
def train_task(self, job_id: str, folder: str, task_type: str):
    """Run training pipeline for a dataset folder."""
    from backend.app.services.pipeline import run_pipeline

    run_pipeline(folder, task_type, job_id=job_id)
    _notify_backend_storage_sync(job_id, folder)
