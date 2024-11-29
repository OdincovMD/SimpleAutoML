import os
from exception.file_system import IncorrectDatasetFormatError

def determine_task_type(path_dataset):
    '''
    Определяет тип задачи на основе структуры директории датасета.
    
    Типы задач:
    - "классификация" для структуры с поддиректориями по классам.
    - "сегментация" для структуры с папками `images` и `labels`.

    Параметры:
        -----------
        dataset_path: str
            Путь к директории датасета.

    Возвращает:
    -----------
    str
        Тип задачи ("классификация" или "сегментация"), либо None в случае ошибки.
    '''
    if not os.path.exists(path_dataset):
        raise FileNotFoundError(f"Директория {path_dataset} не найдена.")
    top_level_items = os.listdir(path_dataset)
    
    if 'images' in top_level_items and 'labels' in top_level_items:
        images_path = os.path.join(path_dataset, 'images')
        labels_path = os.path.join(path_dataset, 'labels')
        
        if os.listdir(images_path) and os.listdir(labels_path):
            if all(file.endswith(".txt") for file in os.listdir(labels_path)):
                print('Выбрана задача: сегментация')
                return "сегментация"
    
    elif all(
        os.path.isdir(os.path.join(path_dataset, item)) and 
        os.listdir(os.path.join(path_dataset, item))  # Проверяем, что поддиректория не пуста
        for item in top_level_items
    ):
        print('Выбрана задача: классификация')
        return "классификация"
    raise IncorrectDatasetFormatError()