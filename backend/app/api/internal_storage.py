"""Внутренний API: синхронизация артефактов обучения в MinIO (только backend общается с хранилищем)."""
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from backend.config import settings
from backend.app.services.storage import sync_task_artifacts_to_minio

router = APIRouter()


class SyncStorageBody(BaseModel):
    job_id: str = Field(..., min_length=1)
    task_folder: str = Field(..., min_length=1)


@router.post("/sync")
def sync_storage_after_train(
    body: SyncStorageBody,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
):
    expected = (settings.INTERNAL_STORAGE_TOKEN or "").strip()
    if not expected or (x_internal_token or "").strip() != expected:
        raise HTTPException(status_code=403, detail="Forbidden")
    sync_task_artifacts_to_minio(body.task_folder, body.job_id)
    return {"ok": True}
