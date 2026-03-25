"""Google Drive API endpoints for dataset selection."""
from fastapi import APIRouter, HTTPException

from backend.app.services.drive import (
    is_valid_drive_file_id,
    list_folders_by_parent_id,
    list_folders_by_parent_name,
    list_root_folders,
)

router = APIRouter()


@router.get("/folders")
def get_folders(parent_name: str | None = None, parent_id: str | None = None) -> list[dict]:
    """
    List dataset folders from Google Drive.
    - parent_name: find folder by name, list its subfolders
    - parent_id: list subfolders of this folder
    - neither: list folders in DRIVE_FOLDER_ID (root)
    """
    try:
        if parent_name is not None and parent_name.strip():
            pn = parent_name.strip()
            if len(pn) > 256:
                raise HTTPException(400, "Слишком длинное имя папки")
            folders = list_folders_by_parent_name(pn)
        elif parent_id is not None and parent_id.strip():
            if not is_valid_drive_file_id(parent_id):
                raise HTTPException(400, "Некорректный parent_id")
            folders = list_folders_by_parent_id(parent_id)
        else:
            folders = list_root_folders()
        return folders
    except FileNotFoundError:
        raise HTTPException(
            503,
            "Google Drive не настроен. Добавьте automl_token.json (service account).",
        )
    except Exception as e:
        raise HTTPException(500, str(e))
