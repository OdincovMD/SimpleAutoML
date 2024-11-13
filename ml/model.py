from ultralytics import YOLO
from ml.check_imgsz import Finder
import torch
import os
import shutil

class Model():
    def __init__(self, model_type, dataset_path, folder, model_path=None, imgsz=None):
        self.model_type = model_type
        self.dataset_path = dataset_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.save_dir = 'train_classify'
        self.folder = folder
        self.imgsz = imgsz
        self.model_path = model_path  # Инициализация атрибута для пути к модели

    def train(self):
        self.imgsz = Finder(self.dataset_path).check_imgsz('', 'yolo11n-cls.yaml', epochs=3)
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

        # Путь к весам модели и папке сохранения
        source_weights_path = os.path.join(self.save_dir, 'train', 'weights', 'best.pt')
        destination_folder = os.path.join('models', self.folder)
        
        os.makedirs(destination_folder, exist_ok=True)

        self.model_path = os.path.join(destination_folder, os.path.basename(source_weights_path))
        shutil.copy2(source_weights_path, self.model_path)

        # Копирование файла с результатами
        source_results_path = os.path.join(self.save_dir, 'train', 'results.png')
        destination_results_path = os.path.join(destination_folder, os.path.basename(source_results_path))
        
        shutil.copy2(source_results_path, destination_results_path)

        # Удаление временной папки с результатами обучения
        shutil.rmtree(self.save_dir, ignore_errors=True)
    