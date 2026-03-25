from fastapi import APIRouter

from celery.result import AsyncResult
from backend.app.tasks import celery_app

router = APIRouter()


def _iso_utc(dt) -> str | None:
    if dt is None:
        return None
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)


@router.get("/{job_id}/status")
def job_status(job_id: str):
    result = AsyncResult(job_id, app=celery_app)
    status = result.status
    error = None
    progress = None
    completed_at = None

    if result.failed():
        err = result.result
        error = str(err) if err is not None else None
    elif result.successful():
        res = result.result
        if isinstance(res, dict):
            progress = res
    else:
        info = result.info
        if isinstance(info, dict):
            progress = info

    if status in ("SUCCESS", "FAILURE", "REVOKED"):
        completed_at = _iso_utc(getattr(result, "date_done", None))

    return {
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "result": None,
        "error": error,
        "completed_at": completed_at,
    }
