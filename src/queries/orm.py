from src.models import Base, DatasetOrm, ModelsOrm
from src.database import session_factory, sync_engine
from sqlalchemy import select,and_
from sqlalchemy.exc import IntegrityError

class SyncOrm:
    @staticmethod
    def create_tables():
        Base.metadata.drop_all(sync_engine)
        Base.metadata.create_all(sync_engine)

    @staticmethod
    def insert_data(row):
        file = DatasetOrm(folder=row['train_folder'], path=row['path'], trained_flag=False)
        with session_factory() as session:
            try:
                session.add(file)
                session.flush()  # Отправляет данные в базу данных, но не сохраняет окончательно
                session.commit()
            except IntegrityError:
                session.rollback()  # Отмена изменений при нарушении уникального ограничения
                
    @staticmethod
    def select_data(folder):
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
        with session_factory() as session:
            session.query(DatasetOrm).filter_by(folder=folder).update({"trained_flag": True})
            session.commit()
    
    @staticmethod
    def insert_model(row):
        file = ModelsOrm(train_folder=row['train_folder'], model_path=row['path'], classes=row['classes'])
        with session_factory() as session:
            session.add(file)
            session.flush()  # Отправляет данные в базу данных, но не сохраняет окончательно
            session.commit()

    @staticmethod
    def select_model(folder):
        with session_factory() as session:
            query = (
                select(ModelsOrm.model_path, ModelsOrm._classes)
                .select_from(ModelsOrm)
                .where(ModelsOrm.train_folder == folder)
            )
            result = session.execute(query)
            return result.fetchall()

    @staticmethod
    def update_model(folder, model_path):
        with session_factory() as session:
            session.query(ModelsOrm).filter_by(ModelsOrm.train_folder == folder).update({'model_path': model_path})
            session.commit()
    


