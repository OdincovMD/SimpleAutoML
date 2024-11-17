from ultralytics import YOLO
from check_imgsz import check_imgsz
from seed import set_seed
import cv2
import torch
import numpy as np
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
        self.test_path = os.path.join(os.path.split(os.path.split(dataset_path)[0])[0], 'test')
        self.result_path = os.path.join(os.path.split(dataset_path)[0], 'results')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.save_dir = r'D:\Mine\Basics\Coding\Python_projects\SimpleAutoML\temp' #'train_classify'
        self.folder = folder
        self.imgsz = imgsz
        self.model_path = model_path  # Инициализация атрибута для пути к модели
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
        self.imgsz = check_imgsz(dataset_path=self.dataset_path, model_type=self.model_type)

        # Инициализация и запуск обучения модели
        model = YOLO(self.model_type)
        model.train(
            data=self.dataset_path,
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
        Дополнительное обучение модели на нвоых данных.
        Результаты и веса копируются в указанную папку.
        """
        model = YOLO(self.model_path)
        model.train(
            epochs=4,
            data=self.dataset_path,
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

        Параметры:
            path_to_images (str): путь до директории, содержащей тестовые данные
            path_to_results (str): путь до директории для сохранения полученных результатов
        """
        # Очистка предыдущих результатов обработки для избежания ошибок, и создание "чистой" директории
        shutil.rmtree(self.result_path, ignore_errors=True)
        os.makedirs(self.result_path)

        for path_to_image in os.listdir(self.test_path):
            abs_path_to_image = os.path.join(self.test_path, path_to_image)
            self._process_image(abs_path_to_image, self.result_path)
        

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

    def _process_image(self, path_to_image: str, path_to_results: str):
        """
        Вспомогательный метод для обработки изображения в зависимости от задачи.
        Параметры:
            path_to_image (str): путь до тестируемого изображения
            path_to_result (str): путь до директории для сохранения полученных результатов
        """
        if 'seg' in self.model_type:
            if not os.path.exists(os.path.join(path_to_results, 'masks')):
                os.makedirs(os.path.join(path_to_results, 'masks'))

            model = YOLO(self.model_path)

            image = cv2.imread(path_to_image)
            image_orig = image.copy()
            h_or, w_or = image.shape[:2]
            image = cv2.resize(image, (640, 640))
            results = model(image)[0]
            
            classes_names = results.names
            classes = results.boxes.cls.cpu().numpy()
            masks = results.masks.data.cpu().numpy()

            for i, mask in enumerate(masks):
                color = self.colors[int(classes[i]) % len(self.colors)]
                
                mask_resized = cv2.resize(mask, (w_or, h_or))
                
                color_mask = np.zeros((h_or, w_or, 3), dtype=np.uint8)
                color_mask[mask_resized > 0] = color

                mask_filename = os.path.join(path_to_results, 'masks', os.path.basename(path_to_image), f"{classes_names[classes[i]]}_{i}.png")
                cv2.imwrite(mask_filename, color_mask)

                image_orig = cv2.addWeighted(image_orig, 1.0, color_mask, 0.5, 0)

            new_image_path = os.path.join(path_to_results, os.path.basename(path_to_image))
            new_image_path = os.path.splitext(new_image_path)[0] + '_yolo' + os.path.splitext(new_image_path)[1]
            cv2.imwrite(new_image_path, image_orig)
        else:
            model = YOLO(self.model_path)
            image = cv2.imread(path_to_image)
            results = model(image)[0]
            print(results.probs)

if __name__ == '__main__':
    shutil.rmtree(r'D:\Mine\Basics\Coding\Python_projects\SimpleAutoML\runs')
    model = Model(r'D:\Mine\Basics\Coding\Python_projects\SimpleAutoML\data_root\Коля\Pothole\dataset\data.yaml',
                r'D:\Mine\Basics\Coding\Python_projects\SimpleAutoML\temp',
                'yolo11n-seg.pt', r'D:\Mine\Basics\Coding\Python_projects\SimpleAutoML\temp', None)
    model.train()
    model.predict()