from ultralytics import YOLO
from torch import device
from torch import cuda
import shutil

def check_imgsz(path_dataset: str, model_type: str, epochs: int = 10) -> int:
    """
    Поиск оптимального размера изображения на конкретном датасете для тренировки yolo.

    Параметры:
        dataset_name (str): Путь до датасета
        model_type (str): Выбор модели для проведения обучения. Допускаются модели для классификации и сегментации.
        epochs (int): Количество эпох для обучения.
    Возвращает:
        int. Оптимальный размер imgsz, найденный в процессе обучения.
    """
    context_imgsz = {}
    for img_size in range(640, 960 + 32, 32):
        model = YOLO(model_type)
        results = model.train(
            data=path_dataset,
            imgsz=img_size,
            epochs=epochs,
            project='train_classify',
            batch=4,
            workers=2,
            device=device("cuda:0" if cuda.is_available() else "cpu")
            )
        metrics = model.val()
        if 'cls' in model_type:
            context_imgsz[img_size] = metrics.top1
        else:
            context_imgsz[img_size] = metrics.box.map
        del model
        if cuda.is_available():
            cuda.empty_cache()
    result = int(max(context_imgsz.items(), key=lambda x: x[1])[0])
    shutil.rmtree('train_classify')
    return result