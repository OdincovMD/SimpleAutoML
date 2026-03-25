import os
import re
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


def iter_minio_object_chunks(bucket: str, object_name: str, chunk_size: int = 65536):
    """
    Потоковое чтение объекта из MinIO (для StreamingResponse через backend, без presigned URL).
    """
    client = get_minio_client()
    response = client.get_object(bucket, object_name)
    try:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            yield chunk
    finally:
        try:
            response.close()
            response.release_conn()
        except Exception:
            pass


def _last_pt_version(object_key: str) -> int | None:
    base = os.path.basename(object_key)
    m = re.match(r"^last_(\d+)\.pt$", base)
    return int(m.group(1)) if m else None


def resolve_model_weights_for_download(folder_id: str, db_version: int) -> tuple[str, str]:
    """
    (bucket, object_key) для presigned GET.
    Сначала точное имя last_{db_version}.pt; иначе любой last_*.pt с максимальной версией.
    Ищет в models (актуально) и results (старый sync до выравнивания бакетов).
    """
    ensure_buckets()
    client = get_minio_client()
    exact_candidates = (
        f"models_{folder_id}/{folder_id}/last_{db_version}.pt",
        f"models_{folder_id}/last_{db_version}.pt",
    )
    for exact in exact_candidates:
        for bucket in ("models", "results"):
            if not client.bucket_exists(bucket):
                continue
            try:
                client.stat_object(bucket, exact)
                return bucket, exact
            except Exception:
                continue

    # Любой last_*.pt под префиксом (вложенный или плоский путь после sync)
    prefix = f"models_{folder_id}/"
    best: tuple[str, str, int] | None = None
    for bucket in ("models", "results"):
        if not client.bucket_exists(bucket):
            continue
        try:
            for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
                if getattr(obj, "is_dir", False):
                    continue
                key = obj.object_name
                if not key.startswith(prefix):
                    continue
                v = _last_pt_version(key)
                if v is None:
                    continue
                if v == db_version:
                    return bucket, key
                if best is None or v > best[2]:
                    best = (bucket, key, v)
        except Exception:
            continue

    if best is not None:
        return best[0], best[1]

    raise FileNotFoundError(
        f"Нет last_*.pt с префиксом {prefix!r} в бакетах models/results"
    )


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
    ensure_buckets()
    client = get_minio_client()

    results_path = os.path.join(task_folder, "results")
    if os.path.isdir(results_path):
        prefix = task_folder.replace("/data/", "").replace(os.sep, "_")
        _fput_tree(client, results_path, prefix, "results")

    # Веса: cwd при обучении — родитель task_dir; YOLO пишет в models/{folder_id}/.
    # Если dataset в подпапке, каталог models может быть только под task_folder.
    for models_path in (
        os.path.join(job_root, "models"),
        os.path.join(task_folder, "models"),
    ):
        ap = os.path.abspath(models_path)
        if os.path.isdir(ap):
            _fput_tree(client, ap, f"models_{folder_id}", "models")

    # Актуальный снимок задачи (в т.ч. новые файлы после дообучения)
    if os.path.isdir(task_folder):
        for f in Path(task_folder).rglob("*"):
            if f.is_file():
                rel = f.relative_to(Path(task_folder))
                obj = f"datasets/{folder_id}/{rel.as_posix()}"
                client.fput_object("datasets", obj, str(f))


def restore_dataset_tree_from_minio(folder_id: str, dest_folder: str) -> bool:
    """Скачать datasets/{folder_id}/** в dest_folder. True если что-то скачано."""
    ensure_buckets()
    client = get_minio_client()
    prefix = f"datasets/{folder_id}/"
    dest_abs = os.path.abspath(dest_folder)
    os.makedirs(dest_abs, exist_ok=True)
    n = 0
    for obj in client.list_objects("datasets", prefix=prefix, recursive=True):
        if getattr(obj, "is_dir", False):
            continue
        key = obj.object_name
        if not key.startswith(prefix):
            continue
        rel = key[len(prefix) :]
        if not rel or rel.endswith("/"):
            continue
        dest = os.path.join(dest_abs, rel)
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        client.fget_object("datasets", key, dest)
        n += 1
    return n > 0


def restore_models_tree_from_minio(folder_id: str, job_root: str) -> bool:
    """Скачать models/models_{folder_id}/** в job_root/models/ (бакеты models и results)."""
    ensure_buckets()
    client = get_minio_client()
    prefix = f"models_{folder_id}/"
    n = 0
    for bucket in ("models", "results"):
        if not client.bucket_exists(bucket):
            continue
        for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
            if getattr(obj, "is_dir", False):
                continue
            key = obj.object_name
            if not key.startswith(prefix):
                continue
            rel = key[len(prefix) :]
            if not rel:
                continue
            dest = os.path.join(job_root, "models", rel)
            parent = os.path.dirname(dest)
            if parent:
                os.makedirs(parent, exist_ok=True)
            client.fget_object(bucket, key, dest)
            n += 1
        if n > 0:
            break
    return n > 0


def upload_inference_zip_to_results(
    local_zip_path: str, folder_id: str, inference_id: str
) -> str:
    """Загрузить архив результатов инференса. Возвращает ключ объекта."""
    ensure_buckets()
    client = get_minio_client()
    key = f"inference/{folder_id}/{inference_id}.zip"
    client.fput_object("results", key, local_zip_path)
    return key
