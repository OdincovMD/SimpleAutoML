import os
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

from backend.app.services.storage import ensure_buckets, get_minio_client
from backend.app.services.drive import download_folder_to
from backend.app.tasks import train_task

router = APIRouter()


class DriveJobRequest(BaseModel):
    folder_id: str


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
            zf.extractall(folder_path)

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
