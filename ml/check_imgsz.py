from ultralytics import YOLO
from torch import device
from torch import cuda
import shutil
import json
import hashlib
import os

# Сетка размеров: 6 точек вместо 11 (640..960 шаг 64)
IMGSZ_GRID = list(range(640, 960 + 1, 64))  # [640, 704, 768, 832, 896, 960]
CACHE_FILE = ".check_imgsz_cache.json"


def _dataset_hash(path_dataset: str, model_type: str) -> str:
    """Хеш датасета и типа модели для кэширования."""
    key = f"{os.path.abspath(path_dataset)}|{model_type}"
    return hashlib.sha256(key.encode()).hexdigest()


def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except IOError:
        pass


def check_imgsz(
    path_dataset: str,
    model_type: str,
    epochs: int = 5,
    use_cache: bool = True,
) -> int:
    """
    Поиск оптимального размера изображения на конкретном датасете для тренировки YOLO.

    Параметры:
        path_dataset (str): Путь до датасета.
        model_type (str): Модель для обучения (классификация или сегментация).
        epochs (int): Количество эпох на каждый размер (по умолчанию 5).
        use_cache (bool): Использовать кэш по хешу датасета (по умолчанию True).

    Возвращает:
        int: Оптимальный размер imgsz.
    """
    cache_key = _dataset_hash(path_dataset, model_type) if use_cache else None

    if use_cache:
        cache = _load_cache()
        if cache_key in cache:
            return int(cache[cache_key])

    context_imgsz = {}
    for img_size in IMGSZ_GRID:
        model = YOLO(model_type)
        model.train(
            data=path_dataset,
            imgsz=img_size,
            epochs=epochs,
            project="train_classify",
            batch=4,
            workers=2,
            device=device("cuda:0" if cuda.is_available() else "cpu"),
            verbose=False,
        )
        metrics = model.val()
        if "cls" in model_type:
            context_imgsz[img_size] = metrics.top1
        else:
            context_imgsz[img_size] = metrics.box.map
        del model
        if cuda.is_available():
            cuda.empty_cache()

    result = int(max(context_imgsz.items(), key=lambda x: x[1])[0])
    shutil.rmtree("train_classify", ignore_errors=True)

    if use_cache:
        cache = _load_cache()
        cache[cache_key] = result
        _save_cache(cache)

    return result