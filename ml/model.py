from ultralytics import YOLO
from ml.check_imgsz import Finder
import torch
import os
import shutil

class Model:
    """
    Класс для обучения и управления YOLO-моделями с дополнительными функциями для сохранения и организации весов модели и результатов.

    Атрибуты:
        model_type (str, optional): Тип используемой модели YOLO
        dataset_path (str): Путь к набору данных, используемому для обучения модели.
        folder (str): Подкаталог для сохранения весов модели и результатов.
        model_path (str, optional): Путь для сохранения или загрузки обученных весов модели. По умолчанию None.
        imgsz (int, optional): Размер изображения для обучения. По умолчанию None.
        device (torch.device): Устройство для обучения ('cuda', если доступно, иначе 'cpu').
        save_dir (str): Каталог, где временно сохраняются результаты обучения и веса модели.
    """

    def __init__(self, dataset_path, folder, model_type=None, model_path=None, imgsz=None):
        self.model_type = model_type
        self.dataset_path = dataset_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.save_dir = 'train_classify'
        self.folder = folder
        self.imgsz = imgsz
        self.model_path = model_path  # Инициализация атрибута для пути к модели

    def train(self):
        """
        Запуск основного этапа обучения модели с параметрами по умолчанию.
        Результаты и веса модели копируются в указанную папку.
        """
        self.imgsz = self.imgsz or 640 ### переделать тут

        # Инициализация и запуск обучения модели
        model = YOLO(self.model_type)
        model.train(
            data=self.dataset_path,
            epochs=2,
            batch=2,
            device=self.device,
            workers=4,
            project=self.save_dir,
            imgsz=self.imgsz
        )

        # Сохранение весов модели и результатов
        self._save_results()

    def additional_train(self):
        """
        Дополнительное обучение модели на нвоых данных.
        Результаты и веса копируются в указанную папку.
        """
        model = YOLO(self.model_path)
        model.train(
            epochs=4,
            data=self.dataset_path,
            batch=2,
            device=self.device,
            workers=4,
            project=self.save_dir,
            imgsz=self.imgsz
        )

        # Сохранение весов модели и результатов
        self._save_results()

    def _save_results(self):
        """
        Вспомогательный метод для сохранения весов и результатов модели.
        """
        source_weights_path = os.path.join(self.save_dir, 'train', 'weights', 'last.pt')
        destination_folder = os.path.join('models', self.folder)
        
        os.makedirs(destination_folder, exist_ok=True)

        self.model_path = os.path.join(destination_folder, os.path.basename(source_weights_path))
        shutil.copy2(source_weights_path, self.model_path)

        source_results_path = os.path.join(self.save_dir, 'train', 'results.png')
        destination_results_path = os.path.join(destination_folder, os.path.basename(source_results_path))
        shutil.copy2(source_results_path, destination_results_path)

        # Удаление временной папки с результатами обучения
        shutil.rmtree(self.save_dir, ignore_errors=True)
