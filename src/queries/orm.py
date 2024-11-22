from src.models import Base, DatasetOrm, ModelsOrm
from src.database import session_factory, sync_engine
from sqlalchemy import select, and_
from sqlalchemy.exc import IntegrityError

class SyncOrm:
    @staticmethod
    def create_tables():
        """Создание и сброс всех таблиц в базе данных."""
        Base.metadata.drop_all(sync_engine)
        Base.metadata.create_all(sync_engine)

    @staticmethod
    def insert_data(row):
        """Вставка данных о наборе данных в базу данных."""
        file = DatasetOrm(folder=row['train_folder'], path=row['path'], trained_flag=False)
        with session_factory() as session:
            try:
                session.add(file)
                session.flush()  # Отправка данных в базу, но без окончательной фиксации
                session.commit()
            except IntegrityError:
                session.rollback()  # Откат изменений при нарушении уникального ограничения

    @staticmethod
    def select_data(folder):
        """Выборка данных по папке."""
        with session_factory() as session:
            query = (
                select(DatasetOrm.path)
                .select_from(DatasetOrm).filter(
                    DatasetOrm.folder == folder
                )
            )
            result = session.execute(query)
            return result.fetchall()

    @staticmethod
    def select_data_not_trained(folder):
        """Выборка данных по папке, где флаг trained_flag равен False."""
        with session_factory() as session:
            query = (
                select(DatasetOrm.path)
                .select_from(DatasetOrm).filter(and_(
                    DatasetOrm.folder == folder,
                    DatasetOrm.trained_flag == False
                ))    
            )
            result = session.execute(query)
            return result.fetchall()

    @staticmethod
    def update_data(folder):
        """Обновление флага trained_flag на True для определенной папки."""
        with session_factory() as session:
            session.query(DatasetOrm).filter_by(folder=folder).update({"trained_flag": True})
            session.commit()
    
    @staticmethod
    def insert_model(row):
        """Вставка данных о модели в базу данных."""
        file = ModelsOrm(train_folder=row['train_folder'], model_path=row['path'], classes=row['classes'], imgsz=row['imgsz'])
        with session_factory() as session:
            session.add(file)
            session.flush()  # Отправка данных в базу, но без окончательной фиксации
            session.commit()

    @staticmethod
    def select_model(folder):
        """Выборка модели по папке обучения."""
        with session_factory() as session:
            query = (
                select(ModelsOrm.model_path, ModelsOrm._classes, ModelsOrm.imgsz)
                .select_from(ModelsOrm)
                .where(ModelsOrm.train_folder == folder)
            )
            result = session.execute(query)
            return result.fetchall()

    @staticmethod
    def update_model(folder, model_path):
        """Обновление пути модели для определенной папки обучения."""
        with session_factory() as session:
            session.query(ModelsOrm).filter(ModelsOrm.train_folder == folder).update({'model_path': model_path})
            session.commit()
