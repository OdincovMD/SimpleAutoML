from ultralytics import YOLO

def check_imgsz(dataset_path: str, model_type: str, epochs: int = 5) -> int:
    """
    Поиск оптимального размера изображения на конкретном датасете для тренировки yolo.

    Параметры:
        dataset_name (str): Путь до датасета
        model_type (str): Выбор модели для проведения обучения. Допускаются модели для классификации и сегментации.
        epochs (int): Количество эпох для обучения. По умолчанию 1000.
    Возвращает:
        int. Оптимальный размер imgsz, найденный в процессе обучения.
    """
    context_imgsz = {}
    for img_size in range(640, 704, 64):
        model = YOLO(model_type)
        results = model.train(
            data=dataset_path,
            imgsz=img_size,
            epochs=epochs,
            batch=4,
            )
        metrics = model.val()
        if 'cls' in model_type:
            context_imgsz[img_size] = metrics.top1
        else:
            context_imgsz[img_size] = metrics.box.map
    result = int(max(context_imgsz.items(), key=lambda x: x[1])[0])
    return result