import yaml
import os
from exception.file_system import FolderError, TxtFileNotFoundError
from ml.classification.augmentation import save_with_augmentations
import random
import shutil
from tqdm import tqdm

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


def spliting_seg(path_to_dataset, train_size=0.9, val_size=0.1, test_size=0.0, shuffle=True, random_seed=42):
    """
    Разделяет датасет на обучающую, валидационную и тестовую выборки и сохраняет их в отдельные папки.

    Аргументы:
    ----------
    path_to_dataset : str
        Путь к корневой директории датасета, содержащей подпапки 'images' и 'labels'.
    
    train_size : float
        Доля данных для обучающей выборки (от 0 до 1).

    val_size : float
        Доля данных для валидационной выборки (от 0 до 1).

    test_size : float, по умолчанию 0
        Доля данных для тестовой выборки (от 0 до 1). Если равно 0, тестовая выборка не создается.

    shuffle : bool, по умолчанию True
        Если True, перемешивает файлы перед разбиением.

    random_seed : int, по умолчанию 42
        Начальное значение для генератора случайных чисел при перемешивании данных.

    Возвращает:
    -----------
    str
        Путь к директории `./dataset`, содержащей подпапки для обучающей, валидационной и (при необходимости) тестовой выборок.
    """

    output_dir = f'./data_root'
    train_image_dir = os.path.join(output_dir, 'train', 'images')
    train_label_dir = os.path.join(output_dir, 'train', 'labels')

    val_image_dir = os.path.join(output_dir, 'val', 'images')
    val_label_dir = os.path.join(output_dir, 'val', 'labels')

    test_image_dir = os.path.join(output_dir, 'test', 'images') if test_size > 0 else None
    test_label_dir = os.path.join(output_dir, 'test', 'labels') if test_size > 0 else None
    
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)

    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    if test_size > 0:
        os.makedirs(test_image_dir, exist_ok=True)
        os.makedirs(test_label_dir, exist_ok=True)

    image_dir = os.path.join(path_to_dataset, 'images')
    label_dir = os.path.join(path_to_dataset, 'labels')
    image_files = sorted(os.listdir(image_dir))
    
    if shuffle:
        random.seed(random_seed)
        random.shuffle(image_files)

    num_files = len(image_files)

    train_end = int(num_files * train_size)
    val_end = train_end + int(num_files * val_size)
    
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:] if test_size > 0 else []

    save_files_to_dir(train_files, image_dir, label_dir, train_image_dir, train_label_dir, desc="Copying train files")
    save_files_to_dir(val_files, image_dir, label_dir, val_image_dir, val_label_dir, desc="Copying val files")

    if test_size > 0:
        save_files_to_dir(test_files, image_dir, label_dir, test_image_dir, test_label_dir, desc="Copying test files")

    return output_dir

def building_yaml(input_folder, output_folder):
    """
    Создает YAML-файл для конфигурации датасета с указанием путей к тренировочным, валидационным и тестовым изображениям.
    
    Функция находит все уникальные первые символы каждой строки во всех текстовых файлах
    в папке "labels" и создает список классов для датасета, запрашивая у пользователя
    наименование каждого уникального класса.

    Параметры:
    - input_folder (str): Путь к папке с исходными данными (содержит папку labels с аннотациями).
    - output_folder (str): Путь к папке, где будет сохранен YAML-файл с конфигурацией датасета.

    Исключения:
    - FolderError: Вызывается, если файл не является текстовым.
    - FileNotFoundError: Вызывается, если папка labels отсутствует в input_folder.
    - ValueError: Вызывается, если уникальные символы в файле не являются числами.

    Возвращает:
    - names: Лист с метками.
    """
    
    unique_chars = set()
    directory = os.path.join(input_folder, 'labels')

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
    names = []
    try:
        for el in sorted(map(int, unique_chars)):
            name = input(f'Класс : {el}. Наименование: ')
            names.append(name)
    except ValueError:
        raise ValueError("Все уникальные символы в аннотациях должны быть числами")
    create_yaml(names, output_folder)
    return names

def create_yaml(names, output_folder):
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
        'train': f'{output_folder}/train/images',
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


def spliting_class(path_to_dataset, val_size=0.85, train_size=0.9, shuffle=False, random_seed=42):
    """
    Загрузка и разбиение набора данных для классификатора на тренировочную, валидационную и тестовую части.
    Также автоматически применяет аугментацию для классов с меньшим количеством примеров.

    Параметры:
        path_to_dataset (str): Путь к директории, содержащей поддиректории с изображениями по классам.
        val_size (float): Доля данных для валидационного набора.
        train_size (float): Доля данных для тренировочного набора (должна быть меньше `val_size`).
        shuffle (bool): Если `True`, перемешивает файлы перед разбиением.
        random_seed (int): Сид для случайной генерации (при shuffle=True).
    Возвращает:
        None. Функция сохраняет файлы в структуре директорий: './data_root/train', './data_root/val', './data_root/test'.
    """
    output_dir = './data_root'
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    class_names = os.listdir(path_to_dataset)

    class_count = {}
    for class_name in class_names:
        class_dir = os.path.join(path_to_dataset, class_name)
        class_count[class_name] = len(os.listdir(class_dir))

    max_class_name, max_class_count = max(class_count.items(), key=lambda item: item[1])

    for class_name in class_names:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        source_dir = os.path.join(path_to_dataset, class_name)
        class_files = os.listdir(source_dir)
        
        if shuffle:
            random.seed(random_seed)
            random.shuffle(class_files)

        split_train_end = int(len(class_files) * val_size)
        split_val_end = int(len(class_files) * train_size)

        train_files = class_files[:split_train_end]
        val_files = class_files[split_train_end:split_val_end]
        test_files = class_files[split_val_end:]
        if class_name != max_class_name:
            augment_factor = max_class_count // class_count[class_name]
        else:
            augment_factor = 0
        save_with_augmentations(train_files, source_dir, train_dir, class_name, desc=f"dir: train | class: {class_name}", augment_factor=augment_factor)
        save_with_augmentations(val_files, source_dir, val_dir, class_name, desc=f"dir: val | class: {class_name}")
        save_with_augmentations(test_files, source_dir, test_dir, class_name, desc=f"dir: test | class: {class_name}")
