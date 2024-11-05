
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.config import settings

sync_engine = create_engine(  # создание синхронного подключения
    url=settings.DATABASE_URL_pymysql,
    pool_size=5, # пять подклкючений к базе данных максимум
    max_overflow=10, # дополнительные подключения, если перебор по подключениям
)


session_factory = sessionmaker(sync_engine)  # фабрика сессий


class Base(DeclarativeBase):
    def __repr__(self):
        cols = [f"{col}={getattr(self, col)}" for col in self.__table__.columns.keys()]
        return f"<{self.__class__.__name__} {','.join(cols)}>"