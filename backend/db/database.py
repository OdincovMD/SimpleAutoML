from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.config import settings

sync_engine = create_engine(
    url=settings.database_url,
    pool_size=5,
    max_overflow=10,
)

session_factory = sessionmaker(sync_engine)


class Base(DeclarativeBase):
    def __repr__(self):
        cols = [f"{col}={getattr(self, col)}" for col in self.__table__.columns.keys()]
        return f"<{self.__class__.__name__} {','.join(cols)}>"
