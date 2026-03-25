"""Загрузка артефактов (результаты инференса) в Google Drive из worker (ml/model.py)."""
from __future__ import annotations

import logging
import os

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from backend.app.services.drive import _escape_drive_query_literal
from backend.config import settings

logger = logging.getLogger(__name__)


def upload_to_drive(filepath: str, drive_path: str) -> None:
    if os.environ.get("SKIP_DRIVE_UPLOAD", "").lower() in ("1", "true", "yes"):
        return
    creds = service_account.Credentials.from_service_account_file(
        settings.SERVICE_ACCOUNT_FILE
    )
    service = build("drive", "v3", credentials=creds)

    folder_id = settings.DRIVE_FOLDER_ID
    for folder_name in drive_path.split(os.sep):
        if not folder_name:
            continue
        logger.info("Drive: обработка папки %s", folder_name)
        safe_name = _escape_drive_query_literal(folder_name)
        query = (
            f"'{folder_id}' in parents and name = '{safe_name}' "
            "and mimeType = 'application/vnd.google-apps.folder'"
        )
        response = (
            service.files()
            .list(q=query, spaces="drive", fields="files(id, name)")
            .execute()
        )
        files = response.get("files", [])

        if not files:
            logger.info("Папка '%s' не найдена, создаём", folder_name)
            file_metadata = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [folder_id],
            }
            folder = service.files().create(body=file_metadata, fields="id").execute()
            folder_id = folder.get("id")
        else:
            folder_id = files[0]["id"]

    file_metadata = {"name": filepath.split(os.sep)[-1], "parents": [folder_id]}
    media = MediaFileUpload(filepath)
    uploaded = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )
    logger.info("Файл загружен в Drive, id=%s", uploaded.get("id"))
