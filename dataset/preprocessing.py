import yaml
import os
from exception.file_system import FolderError

def train_val_split(path_to_dataset):
    '''Функция получает на вход путь до датасета и разбивает его train и val
    Нужно продумать, как это будет происходить:
    1) перекопировать в нужном формате во временную директорию
    2) менять исходную деректорию - плохо
    3) другой вариант
    return: path_to_folder_train_val'''
    return 

def create_yaml(input_folder, output_folder):
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

    data = {
        'train': f'{output_folder}/train/images',
        'val': f'{output_folder}/val/images',
        'test': f'{output_folder}/test/images',
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
    