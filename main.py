import os
import shutil
from dataset.load_dataset import main as load_main_dataset
from dataset.task_selector import determine_task_type
from dataset.splitting import DataSpliting
from dataset.logging import setup_logger
from ml.model import Model
from src.queries.orm import SyncOrm

def train_or_retrain(model_type, split_func, folder, path_dataset):
    """
    Универсальная функция для обучения или дообучения модели.
    """
    if not SyncOrm.select_model(folder):
        data = DataSpliting(path_dataset)
        split_func(data)  # Выполнение сегментации или классификации
        model = Model(model_type=model_type, path_dataset=os.path.join(os.getcwd(), data.output_dir), folder=folder)
        model.train()
        SyncOrm.update_data(folder)
        SyncOrm.insert_model({
            'train_folder': folder,
            'path': model.path_model,
            'classes': data.names,
            'imgsz': model.imgsz
        })
    elif SyncOrm.select_data_not_trained(folder):
        path_model, _, imgsz = SyncOrm.select_model(folder)[0]
        data = DataSpliting(path_dataset)
        split_func(data)
        model = Model(path_model=path_model, path_dataset=os.path.join(os.getcwd(), data.output_dir), folder=folder, imgsz=imgsz)
        model.additional_train()
        SyncOrm.update_data(folder)
        SyncOrm.update_model(folder, model.path_model)

def perform_inference(model, task, folder, path_test):
    """
    Инференс модели на тестовых данных.
    """
    path_model, _, imgsz = SyncOrm.select_model(folder)[0]
    model = Model(path_model=path_model, path_dataset=path_test, folder=folder, imgsz=imgsz)
    model.predict(task)

def main():
    # Создаем таблицы, если это необходимо
    # SyncOrm.create_tables()
    folder = load_main_dataset()
    path_dataset = os.path.join(folder, 'dataset')
    data_root = 'data_root'
    path_test = os.path.join(folder, 'test')
    for root, _, files in os.walk(folder):
        if os.path.basename(root) not in ['test', 'results', 'masks']:
            for file in files:
                SyncOrm.insert_data({'train_folder': folder, 'path': os.path.join(root, file)})
    task = determine_task_type(path_dataset=path_dataset)
    if task == 'сегментация':
        train_or_retrain('yolo11m-seg.pt', lambda data: data.spliting_seg(), folder, path_dataset)
        model = SyncOrm.select_model(folder)[0][0] if SyncOrm.select_model(folder) else None
        if model and input('Хотите ли провести тестирование (Y/N): ') == 'Y':
            perform_inference(model, task, folder, path_test)
    
    elif task == 'классификация':
        train_or_retrain('yolo11m-cls.pt', lambda data: data.spliting_class(), folder, path_dataset)
        model = SyncOrm.select_model(folder)[0][0] if SyncOrm.select_model(folder) else None
        if model and input('Хотите ли провести тестирование (Y/N): ') == 'Y':
            perform_inference(model, task, folder, path_test)
    
    # Удаление временных данных
    if os.path.exists(data_root):
        shutil.rmtree(data_root)

if __name__ == "__main__":
    try:
        setup_logger()
        main()
    except Exception as e:
        print(f"Произошла ошибка: {e}")