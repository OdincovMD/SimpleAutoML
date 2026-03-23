"""Google Drive API endpoints for dataset selection."""
from fastapi import APIRouter, HTTPException

from backend.app.services.drive import (
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
        if parent_name:
            folders = list_folders_by_parent_name(parent_name)
        elif parent_id:
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
