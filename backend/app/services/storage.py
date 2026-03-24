import os
from pathlib import Path

from minio import Minio
from backend.config import settings


_client: Minio | None = None


def get_minio_client() -> Minio:
    global _client
    if _client is None:
        _client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
    return _client


BUCKETS = ("datasets", "models", "results")


def ensure_buckets() -> None:
    client = get_minio_client()
    for name in BUCKETS:
        if not client.bucket_exists(name):
            client.make_bucket(name)


def get_presigned_url(bucket: str, object_name: str, expires_seconds: int = 3600) -> str:
    client = get_minio_client()
    return client.presigned_get_object(bucket, object_name, expires=expires_seconds)


def _fput_tree(client: Minio, local_path: str, minio_prefix: str, bucket: str) -> None:
    """Загрузить файл или каталог в bucket с префиксом (как раньше в pipeline)."""
    if os.path.isfile(local_path):
        name = os.path.basename(local_path)
        client.fput_object(bucket, f"{minio_prefix}/{name}", local_path)
        return
    base = Path(local_path)
    for f in base.rglob("*"):
        if f.is_file():
            rel = f.relative_to(base)
            obj = f"{minio_prefix}/{rel.as_posix()}"
            client.fput_object(bucket, obj, str(f))


def sync_task_artifacts_to_minio(task_folder: str, job_id: str) -> None:
    """
    После обучения: залить results/ и models/ в MinIO.
    Вызывается только из backend (тот же том ML_DATA_PATH, что и у worker).
    """
    task_folder = os.path.abspath(task_folder)
    job_root = os.path.dirname(task_folder)
    folder_id = job_id
    client = get_minio_client()
    bucket = "results"

    results_path = os.path.join(task_folder, "results")
    if os.path.isdir(results_path):
        prefix = task_folder.replace("/data/", "").replace(os.sep, "_")
        _fput_tree(client, results_path, prefix, bucket)

    models_path = os.path.join(job_root, "models")
    if os.path.isdir(models_path):
        _fput_tree(client, models_path, f"models_{folder_id}", bucket)
