from sqlalchemy import VARCHAR, Index, Boolean, MetaData, UniqueConstraint, Text, INTEGER
import json
from src.database import Base
from sqlalchemy.orm import Mapped, mapped_column
from typing import Annotated

metadata_obj = MetaData()

# Конструкторы типов для упрощения определения колонок
intpk = Annotated[int, mapped_column(primary_key=True)]  # Целочисленный первичный ключ
intk = Annotated[int, mapped_column(INTEGER)]           # Обычная целочисленная колонка
strmy = Annotated[str, mapped_column(VARCHAR(200))]     # Строковая колонка длиной до 200 символов
boolmy = Annotated[bool, mapped_column(Boolean)]        # Булевая колонка

class DatasetOrm(Base):
    """
    Модель таблицы 'database', содержащей информацию о датасетах.

    Атрибуты:
        id (int): Уникальный идентификатор записи. Первичный ключ.
        folder (str): Имя папки, связанной с датасетом.
        path (str): Путь к датасету.
        trained_flag (bool): Флаг, указывающий, была ли выполнена тренировка для этого датасета.
    
    Индексы:
        folder_index: Индекс для ускорения поиска по колонке 'folder'.

    Ограничения:
        unique_folder_path_constraint: Ограничение на уникальность комбинации folder и path.
    """
    __tablename__ = "database"

    id: Mapped[intpk]
    folder: Mapped[strmy]
    path: Mapped[strmy]
    trained_flag: Mapped[boolmy]

    __table_args__ = (
        Index("folder_index", "folder"),
        UniqueConstraint("folder", "path", name="unique_folder_path_constraint")
    )

class ModelsOrm(Base):
    """
    Модель таблицы 'models', содержащей информацию об обученных моделях.

    Атрибуты:
        id (int): Уникальный идентификатор модели. Первичный ключ.
        train_folder (str): Папка, связанная с обучением модели.
        model_path (str): Путь к файлу модели.
        _classes (str): JSON-строка с перечнем классов.
        imgsz (int): Размер входного изображения для модели.
    
    Методы:
        classes (list): Геттер для декодирования JSON-строки классов.
        classes.setter: Сеттер для преобразования списка классов в JSON-строку.
    """
    __tablename__ = "models"

    id: Mapped[intpk]
    train_folder: Mapped[strmy]
    model_path: Mapped[strmy]
    _classes: Mapped[str] = mapped_column("classes", Text, nullable=True)
    imgsz: Mapped[intk]

    @property
    def classes(self) -> list:
        """
        Возвращает список классов, декодированный из JSON-строки.

        Returns:
            list: Список классов.
        """
        return json.loads(self._classes) if self._classes else []

    @classes.setter
    def classes(self, value: list):
        """
        Устанавливает список классов, преобразуя его в JSON-строку.

        Args:
            value (list): Список классов.
        """
        self._classes = json.dumps(value, ensure_ascii=False)
