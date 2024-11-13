from ultralytics import YOLO
import shutil
import os

data_root = r'D:\Mine\Basics\Coding\Python_projects\SimpleAutoML\ml\datasets'
# shutil.rmtree(os.path.join(os.path.join(*(data_root.split("\\")[:-1])), 'segmentation', 'runs'))

class Finder():
    def __init__(self):
        self.checked = {}

    # Allowed model_type: 
    #   "yolo{ver}{size}.pt", "yolo{ver}{size}-seg.pt", "yolo{ver}{size}-cls.pt" 
    # Where
    #   ver = 8, 11
    #   size = n, s, m, l, x
    def check_imgsz(self, dataset_name: str, model_type: str, epochs: int = 150) -> int:
        """
        Поиск оптимального размера изображения на конкретном датасете для тренировки yolo.

        Параметры:
            dataset_name (str): Название конкретного датасета.
            model_type (str): Выбор модели для проведения обучения. Допускаются модели для классификации и сегментаци.
            epochs (int): Количество эпох для обучения. По умолчанию 1000.
        Возвращает:
            int. Оптимальный размер imgsz, найденный в процессе обучения.
        """
        context_imgsz = {}
        for img_size in range(640, 1312, 64):
            model = YOLO(model_type)
            results = model.train(
                data=os.path.join(data_root, dataset_name, r'data.yaml'),
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
        self.checked[dataset_name] = result
        return self.checked[dataset_name]

# seg = Finder()
# seg.check_imgsz('Butterfly', model_type='yolo11n-seg.pt', epochs=150)