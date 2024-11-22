import yaml
import os
from exception.file_system import FolderError, TxtFileNotFoundError, NotEnoughImagesError
from ml.augmentation import save_with_augmentations
from ml.seed import set_seed
import random
import shutil
from tqdm import tqdm

class DataSpliting():
    """
    Класс DataSpliting

    Класс предназначен для организации и разделения датасетов на обучающую, валидационную и тестовую выборки.
    Включает функции для копирования и аугментации пар изображений и меток, создания YAML-конфигурационного файла для классов 
    датасета, а также для управления датасетами с директориями, организованными по классам.

    Инициализация
    -------------
    def __init__(self, path_to_dataset, random_seed=42, shuffle=False)

    Параметры:
    - path_to_dataset (str): Путь к директории с датасетом.
    - random_seed (int): Случайное зерно для шифрования и воспроизводимости. По умолчанию 42.
    - shuffle (bool): Перемешивать ли файлы перед разделением. По умолчанию False.

    Методы
    ------
    1. save_files_to_dir(files, image_dir, label_dir, dest_image_dir, dest_label_dir, desc)
    Копирует изображения и соответствующие текстовые файлы меток из исходных директорий в целевые.
    
    2. spliting_seg(train_size=0.9, val_size=0.1, test_size=0.0)
    Разделяет датасет на обучающую, валидационную и тестовую выборки и сохраняет в отдельные папки.

    3. building_yaml()
    Создает YAML-файл конфигурации для датасета, указывая пути к обучающим, валидационным и тестовым изображениям.
    Функция находит уникальные символы в текстовых файлах меток и создает список классов, запрашивая у пользователя их названия.

    4. create_yaml(names, output_folder)
    Создает YAML-файл конфигурации датасета с указанными именами классов.

    5. copy_files(new_train_paths, destination_folder)
    Копирует файлы из указанных путей в целевую директорию.

    6. spliting_class(train_size, val_size)
    Разделяет данные по классам на обучающую и валидационную выборки, создавая папки для каждого класса.
    """

    def __init__(self, path_to_dataset, random_seed=42, shuffle=False):
        self.path_to_dataset = path_to_dataset
        self.random_seed = random_seed
        set_seed(self.random_seed)
        self.shuffle = shuffle
    
    @staticmethod
    def save_files_to_dir(files, image_dir, label_dir, dest_image_dir, dest_label_dir, desc):
        """
        Копирует изображения и соответствующие текстовые файлы меток из исходных директорий в целевые.

        Аргументы:
        ----------
        files : list
            Список имен файлов изображений, которые нужно скопировать.
        
        image_dir : str
            Путь к исходной директории с изображениями.
        
        label_dir : str
            Путь к исходной директории с текстовыми файлами меток.
        
        dest_image_dir : str
            Путь к целевой директории для изображений.
        
        dest_label_dir : str
            Путь к целевой директории для текстовых файлов меток.
        
        desc : str
            Описание для отображения прогресса в tqdm.

        Исключения:
        -----------
        TxtFileNotFoundError
            Возникает, если соответствующий файл меток (.txt) для изображения не найден в исходной директории `label_dir`.
        """
        for file in tqdm(files, desc=desc):

            image_path = os.path.join(image_dir, file)
            dest_image_path = os.path.join(dest_image_dir, file)

            label_file = os.path.splitext(file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            dest_label_path = os.path.join(dest_label_dir, label_file)
            try:
                shutil.copy(label_path, dest_label_path)
                shutil.copy(image_path, dest_image_path)

            except Exception:
                raise TxtFileNotFoundError(label_file, label_dir)


    def spliting_seg(self, train_size=0.9, val_size=0.1, test_size=0.0):
        """
        Разделяет датасет на обучающую, валидационную и тестовую выборки и сохраняет их в отдельные папки.

        Аргументы:
        ----------
        train_size : float
            Доля данных для обучающей выборки (от 0 до 1).

        val_size : float
            Доля данных для валидационной выборки (от 0 до 1).

        test_size : float, по умолчанию 0
            Доля данных для тестовой выборки (от 0 до 1). Если равно 0, тестовая выборка не создается.

        Возвращает:
        -----------
        str
            Путь к директории `dataset`, содержащей подпапки для обучающей, валидационной и (при необходимости) тестовой выборок.

        Исключения:
        -----------
        NotEnoughImagesError
            Возникает, если количество изображений в датасете недостаточно для добавления в валидационную выборку хотя бы одного изображения.
        """

        if self._get_valsize(val_size) == 0:
            raise(NotEnoughImagesError(self.path_to_dataset))

        self.output_dir = f'data_root'
        train_image_dir = os.path.join(self.output_dir, 'train', 'images')
        train_label_dir = os.path.join(self.output_dir, 'train', 'labels')

        val_image_dir = os.path.join(self.output_dir, 'val', 'images')
        val_label_dir = os.path.join(self.output_dir, 'val', 'labels')

        test_image_dir = os.path.join(self.output_dir, 'test', 'images') if test_size > 0 else None
        test_label_dir = os.path.join(self.output_dir, 'test', 'labels') if test_size > 0 else None
        
        os.makedirs(train_image_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)

        os.makedirs(val_image_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

        if test_size > 0:
            os.makedirs(test_image_dir, exist_ok=True)
            os.makedirs(test_label_dir, exist_ok=True)

        image_dir = os.path.join(self.path_to_dataset, 'images')
        label_dir = os.path.join(self.path_to_dataset, 'labels')
        image_files = sorted(os.listdir(image_dir))
        
        if self.shuffle:
            random.shuffle(image_files)

        num_files = len(image_files)

        train_end = int(num_files * train_size)
        val_end = train_end + int(num_files * val_size)
        
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:] if test_size > 0 else []

        self.save_files_to_dir(train_files, image_dir, label_dir, train_image_dir, train_label_dir, desc="Copying train files")
        self.save_files_to_dir(val_files, image_dir, label_dir, val_image_dir, val_label_dir, desc="Copying val files")

        if test_size > 0:
            self.save_files_to_dir(test_files, image_dir, label_dir, test_image_dir, test_label_dir, desc="Copying test files")
        self.building_yaml()

    def building_yaml(self):
        """
        Создает YAML-файл для конфигурации датасета с указанием путей к тренировочным, валидационным и тестовым изображениям.
        
        Функция находит все уникальные первые символы каждой строки во всех текстовых файлах
        в папке "labels" и создает список классов для датасета, запрашивая у пользователя
        наименование каждого уникального класса.

        Исключения:
        - FolderError: Вызывается, если файл не является текстовым.
        - FileNotFoundError: Вызывается, если папка labels отсутствует в input_folder.
        - ValueError: Вызывается, если уникальные символы в файле не являются числами.
        """
        
        unique_chars = set()
        directory = os.path.join(self.path_to_dataset, 'labels')

        if not os.path.exists(directory):
            raise FileNotFoundError(f"Директория {directory} не найдена")
        
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        for line in file:
                            line = line.strip()
                            if line:
                                unique_chars.add(line[0])  # Собираем уникальные первые символы
                except Exception as e:
                    raise IOError(f"Ошибка при чтении файла {filepath}: {e}")
            else:
                raise FolderError(filename)

        print('Введите, что означает каждый класс в датасете')
        self.names = []
        try:
            for el in sorted(map(int, unique_chars)):
                name = input(f'Класс : {el}. Наименование: ')
                self.names.append(name)
        except ValueError:
            raise ValueError("Все уникальные символы в аннотациях должны быть числами")
        self.create_yaml(self.names, self.output_dir)
    
    def create_yaml(self, names, output_folder):
        """
        Создает YAML-файл с конфигурацией датасета для проекта машинного обучения.

        Параметры:
        ----------
        names : list of str
            Список имен классов, используемых в датасете. Каждый элемент списка представляет один класс.
            
        output_folder : str
            Путь к папке, в которой будет сохранен YAML-файл. Эта папка должна содержать
            поддиректорию 'train/images' для датасета.

        Возвращает:
        -------
        None

        Исключения:
        ------
        IOError
            Если возникает ошибка записи YAML-файла, поднимается исключение IOError с сообщением об ошибке.
        """
        data = {
            'train': 'train/images',
            'val': f'val/images',
            'nc': len(names),
            'names': names
        }
        output_path = os.path.join(output_folder, 'dataset.yaml')
        try:
            with open(output_path, 'w') as file:
                yaml.dump(data, file, default_flow_style=None, allow_unicode=True)
            print(f"YAML файл успешно создан: {output_path}")
        except Exception as e:
            raise IOError(f"Ошибка при записи YAML-файла: {e}") 
        self.output_dir = output_path
    
    @staticmethod
    def copy_files(new_train_paths, destination_folder):
        """
        Копирует файлы из путей, указанных в списке `new_train_paths`, в указанную папку `destination_folder`.

        Параметры:
        ----------
        new_train_paths : list of str
            Список путей к исходным файлам, которые необходимо скопировать.
        
        destination_folder : str
            Папка, в которую будут скопированы файлы.

        Возвращает:
        -------
        None
        """
        os.makedirs(destination_folder, exist_ok=True)

        for src_path in new_train_paths:
            if os.path.exists(src_path):
                file_name = os.path.basename(src_path)
                dest_path = os.path.join(destination_folder, file_name)
                shutil.copy2(src_path, dest_path)

    def spliting_class(self, train_size=0.9, val_size=0.1):
        """
        Загрузка и разбиение набора данных для классификатора на тренировочную, валидационную и тестовую части.
        Также автоматически применяет аугментацию для классов с меньшим количеством примеров.

        Параметры:
            val_size (float): Доля данных для валидационного набора.
            train_size (float): Доля данных для тренировочного набора (должна быть меньше `val_size`).
        Возвращает:
            None. Функция сохраняет файлы в структуре директорий: 'data_root/train', 'data_root/val', 'data_root/test'.
        """
        self.output_dir = 'data_root'
        train_dir = os.path.join(self.output_dir, 'train')
        val_dir = os.path.join(self.output_dir, 'val')

        self.names = os.listdir(self.path_to_dataset)

        class_count = {}
        for class_name in self.names:
            class_dir = os.path.join(self.path_to_dataset, class_name)
            class_count[class_name] = len(os.listdir(class_dir))

        max_class_name, max_class_count = max(class_count.items(), key=lambda item: item[1])

        for class_name in self.names:
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

            source_dir = os.path.join(self.path_to_dataset, class_name)
            class_files = os.listdir(source_dir)
            
            if self.shuffle:
                random.shuffle(class_files)

            num_files = len(class_files)
            train_end = int(num_files * train_size)
            val_end = train_end + int(num_files * val_size)
            
            train_files = class_files[:train_end]
            val_files = class_files[train_end:val_end]

            if class_name != max_class_name:
                augment_factor = max_class_count // class_count[class_name]
            else:
                augment_factor = 0
            save_with_augmentations(train_files, source_dir, train_dir, class_name, desc=f"dir: train | class: {class_name}", augment_factor=augment_factor)
            save_with_augmentations(val_files, source_dir, val_dir, class_name, desc=f"dir: val | class: {class_name}")

    def _get_valsize(self, val_size):
        '''
        Вспомогательная функция для определения размера валидационной выборки.
        '''
        image_dir = os.path.join(self.path_to_dataset, 'images')
        image_files = os.listdir(image_dir)
        num_files = len(image_files)
        valsize = int(num_files * val_size)
        return valsize