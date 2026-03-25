import json
from datetime import datetime

from sqlalchemy import VARCHAR, Index, Boolean, Text, INTEGER, UniqueConstraint, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from typing import Annotated

from backend.db.database import Base

intpk = Annotated[int, mapped_column(primary_key=True, autoincrement=True)]
intk = Annotated[int, mapped_column(INTEGER)]
strmy = Annotated[str, mapped_column(VARCHAR(200))]
boolmy = Annotated[bool, mapped_column(Boolean)]


class DatasetOrm(Base):
    __tablename__ = "database"

    id: Mapped[intpk]
    folder: Mapped[strmy]
    path: Mapped[strmy]
    trained_flag: Mapped[boolmy]

    __table_args__ = (
        Index("folder_index", "folder"),
        UniqueConstraint("folder", "path", name="unique_folder_path_constraint"),
    )


class ModelsOrm(Base):
    __tablename__ = "models"

    id: Mapped[intpk]
    train_folder: Mapped[strmy]
    model_path: Mapped[strmy]
    version: Mapped[intk]
    _classes: Mapped[str] = mapped_column("classes", Text, nullable=True)
    imgsz: Mapped[intk]
    task_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    trained_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    @property
    def classes(self) -> list:
        return json.loads(self._classes) if self._classes else []

    @classes.setter
    def classes(self, value: list):
        self._classes = json.dumps(value, ensure_ascii=False)
