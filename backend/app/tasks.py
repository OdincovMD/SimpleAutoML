import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from collections import deque
from pathlib import Path

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


def _notify_backend_storage_sync(job_id: str, task_folder: str) -> None:
    """Синхронизация артефактов: worker → backend → MinIO."""
    backend_url = os.environ.get("BACKEND_INTERNAL_URL", "http://backend:8000").rstrip("/")
    token = (os.environ.get("INTERNAL_STORAGE_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("INTERNAL_STORAGE_TOKEN is not set")
    payload = json.dumps({"job_id": job_id, "task_folder": task_folder}).encode()
    req = urllib.request.Request(
        f"{backend_url}/api/internal/storage/sync",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Internal-Token": token,
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=3600)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"Storage sync failed: HTTP {e.code} {body}") from e


def _notify_backend_inference_upload(
    folder_id: str, inference_id: str, zip_path: str
) -> None:
    backend_url = os.environ.get("BACKEND_INTERNAL_URL", "http://backend:8000").rstrip("/")
    token = (os.environ.get("INTERNAL_STORAGE_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("INTERNAL_STORAGE_TOKEN is not set")
    payload = json.dumps(
        {"folder_id": folder_id, "inference_id": inference_id, "zip_path": zip_path}
    ).encode()
    req = urllib.request.Request(
        f"{backend_url}/api/internal/storage/inference",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Internal-Token": token,
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"Inference upload failed: HTTP {e.code} {body}") from e


def _find_folder_with_images(root: str) -> str:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    q = deque([root])
    while q:
        d = q.popleft()
        try:
            names = os.listdir(d)
        except OSError:
            continue
        files = [f for f in names if os.path.isfile(os.path.join(d, f))]
        if any(Path(f).suffix.lower() in exts for f in files):
            return d
        for name in names:
            sub = os.path.join(d, name)
            if os.path.isdir(sub):
                q.append(sub)
    return root


@celery_app.task(bind=True)
def train_task(self, job_id: str, folder: str, task_type: str):
    """Пайплайн обучения; этапы пишутся в meta задачи Celery."""
    from backend.app.services.pipeline import run_pipeline

    steps_history: list[str] = []

    def report(step_id: str) -> None:
        steps_history.append(step_id)
        tail = steps_history[-100:]
        self.update_state(
            state="STARTED",
            meta={
                "kind": "train",
                "step": step_id,
                "steps_history": tail,
                "folder_id": job_id,
                "task_type": task_type,
            },
        )

    report("job_queued")
    run_pipeline(folder, task_type, job_id=job_id, progress_callback=report)
    report("saving_to_cloud")
    _notify_backend_storage_sync(job_id, folder)
    report("job_complete")
    return {
        "kind": "train",
        "folder_id": job_id,
        "task_type": task_type,
        "step": "job_complete",
        "steps_history": steps_history[-100:],
    }


@celery_app.task(bind=True)
def infer_task(
    self,
    folder_id: str,
    task_type: str,
    zip_path: str,
    inference_id: str,
):
    from backend.db.orm import SyncOrm
    from ml.model import Model
    steps_history: list[str] = []

    def report(step_id: str) -> None:
        steps_history.append(step_id)
        self.update_state(
            state="STARTED",
            meta={
                "kind": "inference",
                "step": step_id,
                "steps_history": steps_history[-100:],
                "folder_id": folder_id,
                "task_type": task_type,
                "inference_id": inference_id,
            },
        )

    data_path = os.path.abspath(os.environ.get("ML_DATA_PATH", "/data"))
    os.makedirs(data_path, exist_ok=True)
    os.chdir(data_path)

    report("infer_queued")
    row = SyncOrm.select_model(folder_id)
    if not row:
        raise RuntimeError("Model missing in DB")

    path_model, version, _classes, imgsz, _ = row
    if not os.path.isfile(path_model):
        from backend.app.services.storage import restore_models_tree_from_minio

        if not restore_models_tree_from_minio(folder_id, data_path):
            raise RuntimeError("Model weights not on disk and not in MinIO")

    report("infer_prepare")
    work = os.path.join(data_path, folder_id, "inference_work", inference_id)
    raw = os.path.join(work, "raw")
    os.makedirs(raw, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for m in zf.infolist():
                if m.is_dir():
                    continue
                target = os.path.normpath(os.path.join(raw, m.filename))
                if not (target == raw or target.startswith(raw + os.sep)):
                    raise ValueError("zip slip")
                parent = os.path.dirname(target)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with zf.open(m, "r") as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)

        test_dir = _find_folder_with_images(raw)
        if not os.listdir(test_dir):
            raise RuntimeError("No images in ZIP")

        report("infer_running")
        model = Model(
            path_dataset=test_dir,
            folder=folder_id,
            path_model=path_model,
            imgsz=imgsz,
            version=version,
        )
        model.predict(task_type)

        path_result = model.path_result
        if not os.path.isdir(path_result) or not os.listdir(path_result):
            raise RuntimeError("Inference produced no results")

        report("infer_pack")
        out_zip_base = os.path.join(
            data_path, folder_id, "inference_artifacts", inference_id
        )
        os.makedirs(os.path.dirname(out_zip_base), exist_ok=True)
        shutil.make_archive(out_zip_base, "zip", path_result)
        out_zip = out_zip_base + ".zip"

        report("infer_upload")
        _notify_backend_inference_upload(folder_id, inference_id, out_zip)

        if os.path.exists(out_zip):
            os.remove(out_zip)
        shutil.rmtree(work, ignore_errors=True)
    finally:
        if os.path.isfile(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass

    report("infer_complete")
    return {
        "kind": "inference",
        "folder_id": folder_id,
        "task_type": task_type,
        "inference_id": inference_id,
        "step": "infer_complete",
        "steps_history": steps_history[-100:],
    }
