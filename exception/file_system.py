class FolderError(Exception):
    def __init__(self) -> None:
        super().__init__(f"Указанное название папки не существует. Проверьте правильность названия и наличие папки")

class EmptyFolderError(Exception):
    def __init__(self, folder: str) -> None:
        super().__init__(f"На уровне папки '{folder}' не найдено вложенных папок.")

class DownloadTypeError(Exception):
    def __init__(self) -> None:
        super().__init__(f"Неверный тип загрузки. Доступные варианты: 'drive' или 'zip'.")

class DownloadError(Exception):
    def __init__(self, *args: str) -> None:
        super().__init__(f"Произошла непредвиденная ошибка. Описание: {' '.join(args)}")

class LabelError(Exception):
    def __init__(self, file: str) -> None:
        super().__init__(f"Папка 'label' содержит недопустимый файл: '{file}'.")

class TxtFileNotFoundError(Exception):
    def __init__(self, filename: str, label_dir: str) -> None:
        super().__init__(f"Файл '{filename}' не найден в папке '{label_dir}'. Операция прервана.")

class NotEnoughImagesError(Exception):
    def __init__(self, source: str) -> None:
        super().__init__(f"Папка {source} не содержит достаточного количества изображений для создания валидационной выборки. Операция прервана.")
        
class TaskSelectionError(Exception):
    def __init__(self, task):
        super().__init__(f"Некорректный выбор задачи: {task}. Допустимые значения: 'сегментация' или 'классификация'.")

class NoTestDataError(Exception):
    def __init__(self):
        super().__init__(f"Директория не содержит тестовых изображений. Выполнение завершено.")

