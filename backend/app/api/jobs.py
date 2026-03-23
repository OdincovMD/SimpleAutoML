from fastapi import APIRouter

from celery.result import AsyncResult
from backend.app.tasks import celery_app

router = APIRouter()


@router.get("/{job_id}/status")
def job_status(job_id: str):
    result = AsyncResult(job_id, app=celery_app)
    return {
        "job_id": job_id,
        "status": result.status,
        "result": str(result.result) if result.ready() and result.successful() else None,
        "error": str(result.result) if result.failed() else None,
    }
