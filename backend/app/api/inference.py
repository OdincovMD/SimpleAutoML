import logging
import os
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from minio.error import S3Error

from backend.db.orm import SyncOrm
from backend.app.services.storage import get_minio_client, iter_minio_object_chunks
from backend.app.tasks import infer_task

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/{folder_id}")
async def start_inference(
    folder_id: str,
    file: UploadFile = File(...),
    task_type: str = Form(...),
):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Нужен ZIP-архив с тестовыми изображениями")

    row = SyncOrm.select_model(folder_id)
    if not row:
        raise HTTPException(404, "Модель не найдена для этого folder_id")

    data_path = os.environ.get("ML_DATA_PATH", "/data")
    upload_dir = os.path.join(data_path, folder_id, "inference_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    inference_upload_id = str(uuid.uuid4())
    zip_path = os.path.join(upload_dir, f"{inference_upload_id}.zip")

    content = await file.read()
    with open(zip_path, "wb") as f:
        f.write(content)

    try:
        t = infer_task.delay(folder_id, task_type, zip_path, inference_upload_id)
    except Exception:
        try:
            os.remove(zip_path)
        except OSError as rm_exc:
            logger.warning("Не удалось удалить временный ZIP после сбоя очереди: %s", rm_exc)
        raise

    return {
        "job_id": t.id,
        "folder_id": folder_id,
        "inference_id": inference_upload_id,
        "inference_upload_id": inference_upload_id,
        "task_type": task_type,
    }


@router.get("/{folder_id}/download/{inference_id}")
def download_inference_result(folder_id: str, inference_id: str):
    key = f"inference/{folder_id}/{inference_id}.zip"
    client = get_minio_client()
    try:
        client.stat_object("results", key)
    except S3Error as exc:
        code = getattr(exc, "code", None) or ""
        if code in ("NoSuchKey", "NoSuchBucket", "NotFound"):
            raise HTTPException(404, "Архив результатов не найден в хранилище") from exc
        logger.warning("MinIO stat_object failed: %s", exc)
        raise HTTPException(
            503, "Хранилище временно недоступно, повторите позже"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error checking inference artifact")
        raise HTTPException(
            503, "Не удалось проверить наличие архива"
        ) from exc
    return {
        "download_url": f"/api/inference/{folder_id}/results/{inference_id}",
    }


@router.get("/{folder_id}/results/{inference_id}")
def download_inference_result_file(folder_id: str, inference_id: str):
    """Отдача ZIP с результатами инференса через backend (без presigned MinIO)."""
    key = f"inference/{folder_id}/{inference_id}.zip"
    client = get_minio_client()
    try:
        client.stat_object("results", key)
    except S3Error as exc:
        code = getattr(exc, "code", None) or ""
        if code in ("NoSuchKey", "NoSuchBucket", "NotFound"):
            raise HTTPException(404, "Архив результатов не найден в хранилище") from exc
        logger.warning("MinIO stat_object failed: %s", exc)
        raise HTTPException(
            503, "Хранилище временно недоступно, повторите позже"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error checking inference artifact")
        raise HTTPException(
            503, "Не удалось проверить наличие архива"
        ) from exc
    filename = f"inference_{inference_id}.zip"
    return StreamingResponse(
        iter_minio_object_chunks("results", key),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
