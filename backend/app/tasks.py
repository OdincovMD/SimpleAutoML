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


@celery_app.task(bind=True)
def train_task(self, job_id: str, folder: str, task_type: str):
    """Run training pipeline for a dataset folder."""
    from backend.app.services.pipeline import run_pipeline

    run_pipeline(folder, task_type, job_id=job_id)
