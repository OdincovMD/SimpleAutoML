import os
import shutil
from dataset.load_dataset import main as load_main_dataset
from dataset.splitting import DataSpliting
from ml.model import Model
from src.queries.orm import SyncOrm

# Создаем таблицы, если это необходимо
SyncOrm.create_tables()
folder = load_main_dataset()
dataset_path = os.path.join(folder, 'dataset')
data_root = './data_root'
test_path = os.path.join(folder, 'test')

for root, _, files in os.walk(folder):
    if os.path.basename(root) != 'test':
        for file in files:
            SyncOrm.insert_data({'train_folder': folder, 'path': os.path.join(root, file)})

task = input("Выберите вашу задачу (сегментация | классификация): ")

# Универсальная функция для обучения или дообучения модели
def train_or_retrain(model_type, split_func):
    # Проверка наличия модели для текущей папки
    if not SyncOrm.select_model(folder):
        data = DataSpliting(dataset_path)
        split_func(data)  # Выполнение сегментации или классификации
        model = Model(model_type=model_type, dataset_path=os.path.join(os.getcwd(), data.output_dir), folder=folder)
        model.train()
        SyncOrm.update_data(folder)
        SyncOrm.insert_model({'train_folder': folder, 'path': model.model_path, 'classes': data.names, 'imgsz': model.imgsz})

    # Проверка необходимости дообучения
    elif SyncOrm.select_data_not_trained(folder):
        model_path, _, imgsz = SyncOrm.select_model(folder)[0]
        data = DataSpliting(dataset_path)
        split_func(data)
        model = Model(model_path=model_path, dataset_path=os.path.join(os.getcwd(), data.output_dir), folder=folder, imgsz=imgsz)
        model.additional_train()
        SyncOrm.update_data(folder)
        SyncOrm.update_model(folder, model.model_path)

# Инференс модели на тестовых данных
def perform_inference(model):
    for filename in os.listdir(test_path):
        result = model(os.path.join(test_path, filename))
        print(result)

# Запуск нужного этапа в зависимости от выбранной задачи
if task == 'сегментация':
    train_or_retrain('yolo11m-seg.pt', lambda data: data.spliting_seg())
    model = SyncOrm.select_model(folder)[0][0] if SyncOrm.select_model(folder) else None
    if model:
        perform_inference(model)

elif task == 'классификация':
    train_or_retrain('yolo11m-cls.pt', lambda data: data.spliting_class())
    model = SyncOrm.select_model(folder)[0][0] if SyncOrm.select_model(folder) else None
    if model:
        perform_inference(model)

# Удаление временных данных
if os.path.exists(data_root):
    shutil.rmtree(data_root)
