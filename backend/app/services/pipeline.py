"""
Pipeline service: обучение и дообучение в веб-стеке (Celery worker).
Артефакты на диск (общий том с backend); выгрузка в MinIO — через backend после задачи.
"""
from __future__ import annotations

import os
import shutil
from collections.abc import Callable

from backend.dataset.splitting import DataSpliting
from ml.model import Model
from backend.db.orm import SyncOrm


def run_pipeline(
    folder: str,
    task_type: str,
    job_id: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """
    progress_callback получает код этапа (см. backend.app.job_progress.STEP_ORDER).
    """
    job_root = os.path.dirname(folder)
    folder_id = job_id or os.path.basename(job_root)
    os.chdir(job_root)

    def _p(step_id: str) -> None:
        if progress_callback:
            progress_callback(step_id)

    path_dataset = os.path.join(folder, "dataset")
    path_test = os.path.join(folder, "test")

    _p("indexing_files")
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
        _train_or_retrain(
            "yolo11m-seg.pt",
            _split_seg,
            folder_id,
            path_dataset,
            data_root,
            _p,
            task_type,
        )
    elif task_type == "классификация":
        _train_or_retrain(
            "yolo11m-cls.pt",
            _split_cls,
            folder_id,
            path_dataset,
            data_root,
            _p,
            task_type,
        )

    if os.path.exists(data_root):
        shutil.rmtree(data_root)


def _train_or_retrain(model_type, split_func, folder, path_dataset, data_root, _p, task_type: str):
    train = False
    if not SyncOrm.select_model(folder):
        train = True
        _p("splitting")
        data = DataSpliting(path_dataset)
        split_func(data)
        _p("learning")
        model = Model(
            model_type=model_type,
            path_dataset=os.path.abspath(data.output_dir),
            folder=folder,
        )
        model.train()
        _p("saving_model")
        SyncOrm.update_data(folder)
    elif SyncOrm.select_data_not_trained(folder):
        train = True
        path_model, version, _, imgsz, _ = SyncOrm.select_model(folder)
        _p("fine_tune_split")
        data = DataSpliting(path_dataset)
        split_func(data)
        _p("fine_tuning")
        model = Model(
            path_model=path_model,
            path_dataset=os.path.abspath(data.output_dir),
            folder=folder,
            imgsz=imgsz,
            version=version,
        )
        model.additional_train()
        _p("fine_tune_saved")
    else:
        _p("nothing_to_train")

    if train:
        SyncOrm.update_data(folder)
        SyncOrm.insert_model({
            "train_folder": folder,
            "path": model.path_model,
            "version": model.version,
            "classes": data.names,
            "imgsz": model.imgsz,
            "task_type": task_type,
        })
