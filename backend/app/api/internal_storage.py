"""Внутренний API: синхронизация артефактов обучения в MinIO (только backend общается с хранилищем)."""
import os

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from backend.config import settings
from backend.app.services.storage import (
    sync_task_artifacts_to_minio,
    upload_inference_zip_to_results,
)

router = APIRouter()


class SyncStorageBody(BaseModel):
    job_id: str = Field(..., min_length=1)
    task_folder: str = Field(..., min_length=1)


class InferenceSyncBody(BaseModel):
    folder_id: str = Field(..., min_length=1)
    inference_id: str = Field(..., min_length=1)
    zip_path: str = Field(..., min_length=1)


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


@router.post("/inference")
def sync_inference_artifact(
    body: InferenceSyncBody,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
):
    expected = (settings.INTERNAL_STORAGE_TOKEN or "").strip()
    if not expected or (x_internal_token or "").strip() != expected:
        raise HTTPException(status_code=403, detail="Forbidden")
    data_root = os.path.realpath(settings.ML_DATA_PATH)
    z = os.path.realpath(body.zip_path)
    allowed_root = os.path.realpath(os.path.join(data_root, body.folder_id))
    allowed_prefix = allowed_root + os.sep
    if not (z == allowed_root or z.startswith(allowed_prefix)):
        raise HTTPException(status_code=400, detail="Invalid zip path")
    if not os.path.isfile(z):
        raise HTTPException(status_code=404, detail="Zip not found")
    upload_inference_zip_to_results(z, body.folder_id, body.inference_id)
    try:
        os.remove(z)
    except OSError:
        pass
    return {"ok": True}
