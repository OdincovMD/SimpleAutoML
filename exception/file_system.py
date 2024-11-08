class FolderError(Exception):
    def __init__(self, folder: object) -> None:
        super().__init__(f'Вы ввели несуществующее название папки. Проверьте правильность названия и наличия папки: {folder}')

class EmptyFolder(Exception):
     def __init__(self, folder: object) -> None:
        super().__init__(f"Папок на уровне  {folder} не найдено.")

class DownloadTypeError(Exception):
    def __init__(self, load_type) -> None:
        super().__init__(f"Неверный тип загрузки. Вы ввели: {load_type}. Доступные варианты drive | zip")

class DownladError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(f"Непредвиденная ошибка. Описание {args[0]}")