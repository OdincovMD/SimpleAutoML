from datetime import datetime, timezone

from sqlalchemy import select, and_, update, desc, func
from sqlalchemy.exc import IntegrityError

from backend.db.models import Base, DatasetOrm, ModelsOrm
from backend.db.database import session_factory, sync_engine


class SyncOrm:
    @staticmethod
    def create_tables():
        Base.metadata.create_all(sync_engine)

    @staticmethod
    def init_db():
        Base.metadata.drop_all(sync_engine)
        Base.metadata.create_all(sync_engine)

    @staticmethod
    def insert_data(row):
        file = DatasetOrm(folder=row["train_folder"], path=row["path"], trained_flag=False)
        with session_factory() as session:
            try:
                session.add(file)
                session.flush()
                session.commit()
            except IntegrityError:
                session.rollback()

    @staticmethod
    def select_data(folder):
        with session_factory() as session:
            query = select(DatasetOrm.path).select_from(DatasetOrm).filter(DatasetOrm.folder == folder)
            result = session.execute(query)
            return result.fetchall()

    @staticmethod
    def select_data_not_trained(folder):
        with session_factory() as session:
            query = (
                select(DatasetOrm.path)
                .select_from(DatasetOrm)
                .filter(and_(DatasetOrm.folder == folder, DatasetOrm.trained_flag == False))
            )
            result = session.execute(query)
            return result.fetchall()

    @staticmethod
    def update_data(folder):
        with session_factory() as session:
            stmt = update(DatasetOrm).where(DatasetOrm.folder == folder).values(trained_flag=True)
            session.execute(stmt)
            session.commit()

    @staticmethod
    def insert_model(row):
        file = ModelsOrm(
            train_folder=row["train_folder"],
            model_path=row["path"],
            version=row["version"],
            classes=row["classes"],
            imgsz=row["imgsz"],
            task_type=row.get("task_type"),
            trained_at=datetime.now(timezone.utc),
        )
        with session_factory() as session:
            session.add(file)
            session.flush()
            session.commit()

    @staticmethod
    def select_model(folder):
        with session_factory() as session:
            query = (
                select(
                    ModelsOrm.model_path,
                    ModelsOrm.version,
                    ModelsOrm._classes,
                    ModelsOrm.imgsz,
                    ModelsOrm.task_type,
                )
                .select_from(ModelsOrm)
                .where(ModelsOrm.train_folder == folder)
            )
            result = session.execute(query)
            rows = result.fetchall()
            return rows[-1] if rows else None

    @staticmethod
    def list_models_latest():
        """По одной записи на train_folder — последняя версия."""
        with session_factory() as session:
            rows = (
                session.execute(
                    select(ModelsOrm).order_by(
                        ModelsOrm.train_folder, desc(ModelsOrm.version)
                    )
                )
                .scalars()
                .all()
            )
            seen: set[str] = set()
            out: list[ModelsOrm] = []
            for r in rows:
                if r.train_folder in seen:
                    continue
                seen.add(r.train_folder)
                out.append(r)
            out.sort(
                key=lambda m: (
                    m.trained_at is None,
                    -(m.trained_at.timestamp()) if m.trained_at else 0.0,
                )
            )
            return out

    @staticmethod
    def dataset_stats(folder: str) -> tuple[int, int]:
        """(всего файлов, не помеченных как обученные)."""
        with session_factory() as session:
            total = session.execute(
                select(func.count())
                .select_from(DatasetOrm)
                .where(DatasetOrm.folder == folder)
            ).scalar()
            pending = session.execute(
                select(func.count())
                .select_from(DatasetOrm)
                .where(
                    and_(DatasetOrm.folder == folder, DatasetOrm.trained_flag.is_(False))
                )
            ).scalar()
            return int(total or 0), int(pending or 0)
