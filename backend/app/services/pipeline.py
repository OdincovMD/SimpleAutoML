"""
Pipeline service: wraps main.py logic for web execution.
Артефакты на диск (общий том с backend); выгрузка в MinIO — через backend после задачи.
"""
import os
import shutil

from backend.dataset.task_selector import determine_task_type
from backend.dataset.splitting import DataSpliting
from ml.model import Model
from backend.db.orm import SyncOrm


def run_pipeline(folder: str, task_type: str, job_id: str | None = None) -> None:
    """
    Execute train/or retrain + optional inference.
    folder: path like /data/job_id/task_folder (parent of dataset/)
    task_type: 'сегментация' or 'классификация'
    job_id: optional id for ORM/storage (default: extract from folder)
    """
    job_root = os.path.dirname(folder)
    folder_id = job_id or os.path.basename(job_root)
    os.chdir(job_root)

    path_dataset = os.path.join(folder, "dataset")
    path_test = os.path.join(folder, "test")

    SyncOrm.create_tables()

    for root, _, files in os.walk(folder):
        if os.path.basename(root) not in ("test", "results", "masks"):
            for file in files:
                SyncOrm.insert_data({"train_folder": folder_id, "path": os.path.join(root, file)})

    data_root = os.path.join(os.path.dirname(folder), "data_root")

    def _split_seg(data):
        data.spliting_seg(interactive=False, output_dir=data_root)

    def _split_cls(data):
        data.spliting_class(0.7, 0.3, output_dir=data_root)

    if task_type == "сегментация":
        _train_or_retrain("yolo11m-seg.pt", _split_seg, folder_id, path_dataset, data_root)
    elif task_type == "классификация":
        _train_or_retrain("yolo11m-cls.pt", _split_cls, folder_id, path_dataset, data_root)

    # Загрузка в MinIO выполняется сервисом backend (см. train_task → /api/internal/storage/sync)

    if os.path.exists(data_root):
        shutil.rmtree(data_root)


def _train_or_retrain(model_type, split_func, folder, path_dataset, data_root):
    train = False
    if not SyncOrm.select_model(folder):
        train = True
        data = DataSpliting(path_dataset)
        split_func(data)
        model = Model(
            model_type=model_type,
            path_dataset=os.path.abspath(data.output_dir),
            folder=folder,
        )
        model.train()
        SyncOrm.update_data(folder)
    elif SyncOrm.select_data_not_trained(folder):
        train = True
        path_model, version, _, imgsz = SyncOrm.select_model(folder)
        data = DataSpliting(path_dataset)
        split_func(data)
        model = Model(
            path_model=path_model,
            path_dataset=os.path.abspath(data.output_dir),
            folder=folder,
            imgsz=imgsz,
            version=version,
        )
        model.additional_train()

    if train:
        SyncOrm.update_data(folder)
        SyncOrm.insert_model({
            "train_folder": folder,
            "path": model.path_model,
            "version": model.version,
            "classes": data.names,
            "imgsz": model.imgsz,
        })
