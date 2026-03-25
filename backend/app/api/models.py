from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.db.orm import SyncOrm
from backend.app.services.storage import (
    iter_minio_object_chunks,
    resolve_model_weights_for_download,
)

router = APIRouter()


class ModelListItem(BaseModel):
    train_folder: str
    version: int
    imgsz: int
    task_type: str | None
    classes: list
    trained_at: datetime | None = None


@router.get("", response_model=list[ModelListItem])
def list_models():
    rows = SyncOrm.list_models_latest()
    return [
        ModelListItem(
            train_folder=r.train_folder,
            version=r.version,
            imgsz=r.imgsz,
            task_type=r.task_type,
            classes=r.classes,
            trained_at=r.trained_at,
        )
        for r in rows
    ]


@router.get("/{folder_id}/download")
def download_model(folder_id: str):
    model = SyncOrm.select_model(folder_id)
    if not model:
        raise HTTPException(404, "Model not found")
    return {"download_url": f"/api/models/{folder_id}/weights"}


@router.get("/{folder_id}/weights")
def download_model_weights_stream(folder_id: str):
    model = SyncOrm.select_model(folder_id)
    if not model:
        raise HTTPException(404, "Model not found")
    _path, version, _, _, _ = model
    try:
        bucket, object_name = resolve_model_weights_for_download(folder_id, version)
    except FileNotFoundError:
        raise HTTPException(
            404,
            "Файл весов не найден в MinIO. Дождитесь окончания обучения и синхронизации, "
            "или проверьте, что объект есть в бакетах models / results.",
        )
    filename = f"last_{version}.pt"
    return StreamingResponse(
        iter_minio_object_chunks(bucket, object_name),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
