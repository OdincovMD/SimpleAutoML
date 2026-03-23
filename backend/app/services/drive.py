"""Google Drive API: list folders, download dataset for web pipeline."""
import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from backend.config import settings

SCOPES = ["https://www.googleapis.com/auth/drive.readonly", "https://www.googleapis.com/auth/drive"]


def _get_service():
    creds = service_account.Credentials.from_service_account_file(
        settings.SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


def list_folders_by_parent_name(parent_name: str) -> list[dict]:
    """Find folder by name, return list of subfolders {id, name}."""
    service = _get_service()
    result = service.files().list(
        q=f"name='{parent_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        spaces="drive",
        fields="files(id, name)",
    ).execute()
    files = result.get("files", [])
    if not files:
        return []
    parent_id = files[0]["id"]
    return list_folders_by_parent_id(parent_id)


def list_folders_by_parent_id(parent_id: str) -> list[dict]:
    """List subfolders of a folder. Returns [{id, name}, ...]."""
    service = _get_service()
    result = service.files().list(
        q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
        spaces="drive",
        fields="files(id, name)",
    ).execute()
    return [{"id": f["id"], "name": f["name"]} for f in result.get("files", [])]


def list_root_folders() -> list[dict]:
    """List folders in DRIVE_FOLDER_ID (root). Returns [{id, name}, ...]."""
    return list_folders_by_parent_id(settings.DRIVE_FOLDER_ID)


def _get_all_files_in_folder(service, folder_id: str, prefix: str = "") -> list[dict]:
    """Recursively list all files. Skips 'result' folder."""
    files = []
    result = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, mimeType)",
    ).execute()
    for f in result.get("files", []):
        path = os.path.join(prefix, f["name"]) if prefix else f["name"]
        if f["mimeType"] == "application/vnd.google-apps.folder":
            if f["name"] == "result":
                continue
            files.extend(_get_all_files_in_folder(service, f["id"], path))
        else:
            files.append({"id": f["id"], "path": path})
    return files


def download_folder_to(folder_id: str, dest_path: str) -> None:
    """Download all files from Drive folder to local dest_path (preserving structure)."""
    service = _get_service()
    all_files = _get_all_files_in_folder(service, folder_id)
    os.makedirs(dest_path, exist_ok=True)
    for file_info in all_files:
        local_path = os.path.join(dest_path, file_info["path"])
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        request = service.files().get_media(fileId=file_info["id"])
        with io.FileIO(local_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
