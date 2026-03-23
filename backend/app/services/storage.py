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
