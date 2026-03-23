from fastapi import APIRouter, HTTPException

from backend.db.orm import SyncOrm
from backend.app.services.storage import get_presigned_url

router = APIRouter()


@router.get("/{folder_id}/download")
def download_model(folder_id: str):
    model = SyncOrm.select_model(folder_id)
    if not model:
        raise HTTPException(404, "Model not found")
    path, version, _, _ = model
    bucket = "models"
    object_name = f"models_{folder_id}/{folder_id}/last_{version}.pt"
    try:
        url = get_presigned_url(bucket, object_name)
        return {"download_url": url}
    except Exception:
        raise HTTPException(404, "Model file not in storage")
