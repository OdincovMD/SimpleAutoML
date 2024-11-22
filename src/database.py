from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.config import settings

# Создание синхронного подключения к базе данных
sync_engine = create_engine(
    url=settings.DATABASE_URL_pymysql,  # URL подключения, задается в конфигурации
    pool_size=5,                        # Максимум пять одновременных соединений
    max_overflow=10                     # Дополнительные соединения при превышении лимита
)

# Создание фабрики сессий
session_factory = sessionmaker(sync_engine)

class Base(DeclarativeBase):
    """
    Базовый класс для всех ORM-моделей.

    Методы:
        __repr__: Представляет объект в читаемом виде, выводя значения всех колонок таблицы.
    """
    def __repr__(self):
        cols = [f"{col}={getattr(self, col)}" for col in self.__table__.columns.keys()]
        return f"<{self.__class__.__name__} {','.join(cols)}>"
