class FolderError(Exception):
    def __init__(self) -> None:
        super().__init__(
            "Указанное название папки не существует. Проверьте правильность названия и наличие папки."
        )


class EmptyFolderError(Exception):
    def __init__(self, folder: str) -> None:
        super().__init__(f"На уровне папки '{folder}' не найдено вложенных папок.")


class DownloadTypeError(Exception):
    def __init__(self) -> None:
        super().__init__(
            "Неверный тип загрузки. Доступные варианты: 'drive' или 'zip'."
        )


class DownloadError(Exception):
    def __init__(self, *args: str) -> None:
        super().__init__(f"Произошла непредвиденная ошибка. Описание: {' '.join(args)}")


class LabelError(Exception):
    def __init__(self, file: str) -> None:
        super().__init__(f"Папка 'label' содержит недопустимый файл: '{file}'.")


class TxtFileNotFoundError(Exception):
    def __init__(self, filename: str, label_dir: str) -> None:
        super().__init__(
            f"Файл '{filename}' не найден в папке '{label_dir}'. Операция прервана."
        )


class NotEnoughImagesError(Exception):
    def __init__(self, source: str) -> None:
        super().__init__(
            f"Папка '{source}' не содержит достаточного количества изображений для создания валидационной выборки. Операция прервана."
        )


class NoTestDataError(Exception):
    def __init__(self):
        super().__init__(
            "Директория не содержит тестовых изображений. Выполнение завершено."
        )


class IncorrectDatasetFormatError(Exception):
    def __init__(self, message="Некорректная структура датасета."):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return (
            f"{self.message}\n\n"
            "Ожидаемые структуры:\n"
            "\nКлассификация:\n"
            "dataset/\n"
            "├── class1/         # Директория для изображений первого класса\n"
            "│   ├── image1.jpg\n"
            "│   ├── image2.jpg\n"
            "│   └── ...\n"
            "├── class2/         # Директория для изображений второго класса\n"
            "│   ├── image1.jpg\n"
            "│   ├── image2.jpg\n"
            "│   └── ...\n"
            "└── ...\n"
            "\nСегментация:\n"
            "dataset/\n"
            "├── images/         # Директория с изображениями\n"
            "│   ├── image1.jpg\n"
            "│   ├── image2.jpg\n"
            "│   └── ...\n"
            "├── labels/         # Директория с разметкой\n"
            "│   ├── image1.txt  # Разметка для image1.jpg\n"
            "│   ├── image2.txt\n"
            "│   └── ...\n"
        )
