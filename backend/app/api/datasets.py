import os
import shutil
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

from backend.db.orm import SyncOrm
from backend.app.services.storage import (
    ensure_buckets,
    get_minio_client,
    restore_dataset_tree_from_minio,
    restore_models_tree_from_minio,
)
from backend.app.services.drive import download_folder_to
from backend.app.tasks import train_task

router = APIRouter()


class DriveJobRequest(BaseModel):
    folder_id: str


def _safe_extract_zip(zf: zipfile.ZipFile, dest: str) -> None:
    dest_abs = os.path.abspath(dest)
    for m in zf.infolist():
        if m.is_dir():
            continue
        target = os.path.normpath(os.path.join(dest_abs, m.filename))
        if not (target == dest_abs or target.startswith(dest_abs + os.sep)):
            raise ValueError("Unsafe path in ZIP")
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with zf.open(m, "r") as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)


@router.get("/{folder_id}/meta")
def dataset_meta(folder_id: str):
    total, pending = SyncOrm.dataset_stats(folder_id)
    has_model = SyncOrm.select_model(folder_id) is not None
    task_type = None
    if has_model:
        *_, task_type = SyncOrm.select_model(folder_id)
    return {
        "folder_id": folder_id,
        "files_total": total,
        "files_pending_train": pending,
        "has_model": has_model,
        "task_type": task_type,
    }


@router.post("/{folder_id}/retrain")
async def retrain_dataset(
    folder_id: str,
    file: UploadFile = File(...),
    task_type: str | None = Form(None),
):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Only ZIP archives are accepted")

    row = SyncOrm.select_model(folder_id)
    if not row:
        raise HTTPException(
            404,
            "Сначала обучите модель для этого folder_id, затем можно дообучить",
        )
    _path, _ver, _cls, _imgsz, stored_task = row
    task = (task_type or stored_task or "").strip()
    if not task:
        raise HTTPException(
            400,
            "Не удалось определить тип задачи: укажите поле task_type в форме",
        )
    if task not in ("сегментация", "классификация"):
        raise HTTPException(400, "task_type: сегментация или классификация")

    data_path = os.environ.get("ML_DATA_PATH", "/data")
    folder_path = os.path.join(data_path, folder_id)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    if not os.listdir(folder_path):
        restored = restore_dataset_tree_from_minio(folder_id, folder_path)
        if not restored:
            raise HTTPException(
                404,
                "Нет локальных данных и пусто в MinIO для этого folder_id",
            )

    job_root = data_path
    model_rel = row[0]
    abs_model = (
        model_rel if os.path.isabs(model_rel) else os.path.join(job_root, model_rel)
    )
    if not os.path.isfile(abs_model):
        restore_models_tree_from_minio(folder_id, job_root)

    zip_path = os.path.join(folder_path, "retrain_upload.zip")
    try:
        content = await file.read()
        with open(zip_path, "wb") as f:
            f.write(content)
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extract_zip(zf, folder_path)
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

    task_dir = None
    for root, dirs, _ in os.walk(folder_path):
        if "dataset" in dirs:
            task_dir = root
            break
    if not task_dir:
        raise HTTPException(400, "В архиве или проекте должна быть папка dataset")

    ensure_buckets()
    client = get_minio_client()
    for f in Path(folder_path).rglob("*"):
        if f.is_file():
            rel = f.relative_to(Path(folder_path))
            obj = f"datasets/{folder_id}/{rel.as_posix()}"
            client.fput_object("datasets", obj, str(f))

    t = train_task.delay(folder_id, task_dir, task)
    return {"job_id": t.id, "folder_id": folder_id, "task": task}


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Only ZIP archives are accepted")

    job_id = str(uuid.uuid4())
    data_path = os.environ.get("ML_DATA_PATH", "/data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    folder_path = os.path.join(data_path, job_id)

    os.makedirs(folder_path, exist_ok=True)
    zip_path = os.path.join(folder_path, "upload.zip")

    try:
        content = await file.read()
        with open(zip_path, "wb") as f:
            f.write(content)

        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extract_zip(zf, folder_path)

        # Find folder that contains dataset/
        task_dir = None
        for root, dirs, _ in os.walk(folder_path):
            if "dataset" in dirs:
                task_dir = root
                break
        if not task_dir:
            raise HTTPException(400, "ZIP must contain a 'dataset' folder")

        dataset_path = os.path.join(task_dir, "dataset")
        task = _detect_task(dataset_path)

        # Store in MinIO
        ensure_buckets()
        client = get_minio_client()
        for f in Path(folder_path).rglob("*"):
            if f.is_file():
                rel = f.relative_to(folder_path)
                obj = f"datasets/{job_id}/{rel.as_posix()}"
                client.fput_object("datasets", obj, str(f))

        t = train_task.delay(job_id, task_dir, task)
        return {"job_id": t.id, "folder_id": job_id, "task": task}
    except zipfile.BadZipFile:
        raise HTTPException(400, "Invalid ZIP file")
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


def _detect_task(dataset_path: str) -> str:
    from backend.dataset.task_selector import determine_task_type
    return determine_task_type(dataset_path)


@router.post("/from-drive")
async def start_job_from_drive(body: DriveJobRequest):
    """Скачать датасет из Google Drive и запустить обучение."""
    job_id = str(uuid.uuid4())
    data_path = os.environ.get("ML_DATA_PATH", "/data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    folder_path = os.path.join(data_path, job_id)
    os.makedirs(folder_path, exist_ok=True)

    try:
        download_folder_to(body.folder_id, folder_path)
    except FileNotFoundError:
        raise HTTPException(503, "Google Drive не настроен (automl_token.json)")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Ошибка загрузки с Drive: {e}")

    # Find task_dir (folder containing dataset/)
    task_dir = None
    for root, dirs, _ in os.walk(folder_path):
        if "dataset" in dirs:
            task_dir = root
            break
    if not task_dir:
        raise HTTPException(400, "Папка Drive должна содержать подпапку dataset")

    dataset_path = os.path.join(task_dir, "dataset")
    task = _detect_task(dataset_path)

    ensure_buckets()
    client = get_minio_client()
    for f in Path(folder_path).rglob("*"):
        if f.is_file():
            rel = f.relative_to(folder_path)
            obj = f"datasets/{job_id}/{rel.as_posix()}"
            client.fput_object("datasets", obj, str(f))

    t = train_task.delay(job_id, task_dir, task)
    return {"job_id": t.id, "folder_id": job_id, "task": task}
