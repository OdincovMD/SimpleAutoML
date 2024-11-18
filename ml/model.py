from ultralytics import YOLO
from check_imgsz import check_imgsz
from seed import set_seed
from PIL import Image
import torch
import numpy as np
import os
import shutil

class Model:
    """
    Класс для обучения и управления YOLO-моделями с дополнительными функциями для сохранения и организации весов модели и результатов.

    Атрибуты:
        model_type (str, optional): Тип используемой модели YOLO
        path_dataset (str): Путь к набору данных, используемому для обучения модели.
        folder (str): Подкаталог для сохранения весов модели и результатов.
        path_model (str, optional): Путь для сохранения или загрузки обученных весов модели. По умолчанию None.
        imgsz (int, optional): Размер изображения для обучения. По умолчанию None.
        device (torch.device): Устройство для обучения ('cuda', если доступно, иначе 'cpu').
        save_dir (str): Каталог, где временно сохраняются результаты обучения и веса модели.
    """

    def __init__(self, path_dataset, folder, model_type=None, path_model=None, imgsz=None):
        self.model_type = model_type

        self.path_dataset = path_dataset
        self.path_test = os.path.join(os.path.split(os.path.split(path_dataset)[0])[0], 'test')
        self.path_result = os.path.join(os.path.split(path_dataset)[0], 'results')
        self.path_model = path_model  # Инициализация атрибута для пути к модели

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.save_dir = 'train_classify'
        self.folder = folder
        self.imgsz = imgsz
        self.random_seed = 42
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
            (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
            (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
        ]
        set_seed(self.random_seed)

    def train(self):
        """
        Запуск основного этапа обучения модели с параметрами по умолчанию.
        Результаты и веса модели копируются в указанную папку.
        """
        self.imgsz = check_imgsz(path_dataset=self.path_dataset, model_type=self.model_type)

        # Инициализация и запуск обучения модели
        model = YOLO(self.model_type)
        model.train(
            data=self.path_dataset,
            epochs=2,
            batch=2,
            device=self.device,
            workers=1,
            project=self.save_dir,
            imgsz=self.imgsz,
            seed=self.random_seed
        )

        # Сохранение весов модели и результатов
        self._save_results()

    def additional_train(self):
        """
        Дополнительное обучение модели на новых данных.
        Результаты и веса копируются в указанную папку.
        """
        model = YOLO(self.path_model)
        model.train(
            epochs=4,
            data=self.path_dataset,
            batch=2,
            device=self.device,
            workers=1,
            project=self.save_dir,
            imgsz=self.imgsz,
            seed=self.random_seed
        )
        # Сохранение весов модели и результатов
        self._save_results()

    def predict(self):
        """
        Проверка созданной модели на тестовых данных пользователя.
        """
        # Очистка предыдущих результатов обработки для избежания ошибок, и создание "чистой" директории
        shutil.rmtree(self.path_result, ignore_errors=True)
        os.makedirs(self.path_result)

        if 'seg' in self.model_type:
            os.makedirs(os.path.join(self.path_result, 'masks'))
            for path_image in os.listdir(self.path_test):
                abs_path_image = os.path.join(self.path_test, path_image)
                self._process_image_seg(abs_path_image, self.path_result)
        elif 'cls' in self.model_type:           
            for path_image in os.listdir(self.path_test):
                abs_path_image = os.path.join(self.path_test, path_image)
                self._process_image_cls(abs_path_image, self.path_result)

    def _save_results(self):
        """
        Вспомогательный метод для сохранения весов и метрик модели.
        """
        source_weights_path = os.path.join(self.save_dir, 'train', 'weights', 'last.pt')
        destination_folder = os.path.join('models', self.folder)
        
        os.makedirs(destination_folder, exist_ok=True)

        self.path_model = os.path.join(destination_folder, os.path.basename(source_weights_path))
        shutil.copy2(source_weights_path, self.path_model)

        source_results_path = os.path.join(self.save_dir, 'train', 'results.png')
        destination_results_path = os.path.join(destination_folder, os.path.basename(source_results_path))
        shutil.copy2(source_results_path, destination_results_path)

        # Удаление временной папки с результатами обучения
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def _process_image_seg(self, path_image: str, path_results: str):
        """
        Вспомогательный метод для обработки изображения в задаче сегментации.
        Параметры:
            path_image (str): абсолютный путь до тестируемого изображения
            path_results (str): абсолютный путь до директории для сохранения полученных результатов
        """
        image_name, image_ext = os.path.splitext(os.path.basename(path_image))

        model = YOLO(self.path_model)
        result = model(path_image)[0]

        image_orig = result.orig_img
        h_or, w_or = result.orig_shape

        image = Image.fromarray(image_orig)
        image = image.resize((640, 640))
    
        classes_names = result.names
        classes = result.boxes.cls.cpu().numpy()
        masks = result.masks.data.cpu().numpy()

        for i, mask in enumerate(masks):
            mask = Image.fromarray(mask)            
            mask_resized = mask.resize((w_or, h_or))
            
            color = self.colors[int(classes[i]) % len(self.colors)]
            color_mask = np.zeros((h_or, w_or, 3), dtype=np.uint8)
            color_mask[mask_resized > 0] = color
            color_mask_img = Image.fromarray(color_mask)

            mask_filename = os.path.join(path_results, 'masks', image_name, f"{classes_names[classes[i]]}_{i}{image_ext}")
            color_mask_img.save(mask_filename)

            image_orig = Image.fromarray(image_orig*1.0 + color_mask*0.5)

        new_path_image = os.path.join(path_results, os.path.basename(path_image))
        new_path_image = os.path.splitext(new_path_image)[0] + '_yolo' + os.path.splitext(new_path_image)[1]
        image_orig.save(new_path_image)

    def _process_image_cls(self, path_image: str, path_results: str):
        """
        Вспомогательный метод для обработки изображения в задаче классификации.
        Параметры:
            path_image (str): абсолютный путь до тестируемого изображения
            path_results (str): абсолютный путь до директории для сохранения полученных результатов
        """
        image_name, _ = os.path.splitext(os.path.basename(path_image))

        model = YOLO(self.path_model)
        result = model(path_image)[0]

        classes_names = result.names
        with open(os.path.join(path_results, f"{image_name}_pred.txt"),
                  mode="wt", encoding='utf-8') as pred:
            pred.write({classes_names[result.top1]})

# if __name__ == '__main__':
#     shutil.rmtree(r'D:\Mine\Basics\Coding\Python_projects\SimpleAutoML\runs')
#     model = Model(r'D:\Mine\Basics\Coding\Python_projects\SimpleAutoML\data_root\Коля\Pothole\dataset\data.yaml',
#                 r'D:\Mine\Basics\Coding\Python_projects\SimpleAutoML\temp',
#                 'yolo11n-seg.pt', r'D:\Mine\Basics\Coding\Python_projects\SimpleAutoML\temp', None)
#     model.train()
#     model.predict()