from sqlalchemy import VARCHAR, Index, Boolean, MetaData, UniqueConstraint
from src.database import Base
from sqlalchemy.orm import Mapped, mapped_column
from typing import Annotated

metadata_obj = MetaData()

# конструктор типов для сокращения кода
intpk = Annotated[int, mapped_column(primary_key=True)]
strmy = Annotated[str, mapped_column(VARCHAR(100))]
boolmy = Annotated[bool, mapped_column(Boolean)]

class DatasetOrm(Base):
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
    __tablename__ = "models"

    id: Mapped[intpk]
    train_folder: Mapped[strmy]
    model_path: Mapped[strmy]