import os
import shutil
import dataset.load_dataset as load_dataset
from dataset.spliting import DataSpliting
from ml.model import Model
from src.queries.orm import SyncOrm 

SyncOrm.create_tables()
folder = load_dataset.main()
dataset_path = os.path.join(folder, 'dataset')
data_root = './data_root'

for root, dirs, files in os.walk(folder):
    if os.path.basename(root) != 'test':
        for file in files:
            SyncOrm.insert_data({'train_folder': folder, 'path': os.path.join(root, file)})

if input("Выберите вашу задачу: (сегментация | классификация): ") == 'сегментация':

    # Проверка, существует ли модель для текущей папки
    if not SyncOrm.select_model(folder):
        data = DataSpliting(dataset_path)
        data.spliting_seg()
        # model = Обучение(dataset_path) # Здесь происходит обучение модели
        SyncOrm.update_data(folder)
        # SyncOrm.insert_model({'train_folder': folder, 'path': model_path, 'classes': data.names})

    # Если есть данные для дообучения, дообучаем модель
    elif add_training := SyncOrm.select_data_not_trained(folder):
        model_path, classes = SyncOrm.select_model(folder)[0]
        new_train_paths = [os.path.join(dataset_path, entry[0]) for entry in add_training]
        data = DataSpliting(dataset_path)
        data.copy_files(new_train_paths, destination_folder=data_root)
        data.create_yaml(classes, data_root)
        # model = Дообучение(new_train_paths) # Здесь происходит дообучение модели
        SyncOrm.update_data(folder)
        SyncOrm.update_model(folder, model_path)

    # Если модель уже существует, проводим инференс
    else:
        model = SyncOrm.select_model(folder)[0][0]
        test_path = os.path.join(folder, 'test')
        for filename in os.listdir(test_path):
            result = model(os.path.join(test_path, filename))
            print(result)

else:
    if not SyncOrm.select_model(folder):
        data = DataSpliting(dataset_path)
        data.spliting_class()
        model = Model(model_type='yolo11n-cls.yaml', dataset_path=data.output_dir, folder=folder)
        model.train()
        SyncOrm.update_data(folder)
        SyncOrm.insert_model({'train_folder': folder, 'path': model.model_path, 'classes': data.names, 'imgsz': model.imgsz})

    elif add_training := SyncOrm.select_data_not_trained(folder):
        data_root = './data_root'
        model_path, _ = SyncOrm.select_model(folder)[0]
        new_train_paths = [os.path.join(dataset_path, entry[0]) for entry in add_training]
        data = DataSpliting(dataset_path)
        data.copy_files(new_train_paths, destination_folder=data_root)
        # model = Дообучение(new_train_paths) # Здесь происходит дообучение модели
        SyncOrm.update_data(folder)
        SyncOrm.update_model(folder, model_path)
     # Если модель уже существует, проводим инференс
    else:
        model = SyncOrm.select_model(folder)[0][0]
        test_path = os.path.join(folder, 'test')
        for filename in os.listdir(test_path):
            result = model(os.path.join(test_path, filename))
            print(result)

if os.path.exists(data_root):
        shutil.rmtree(data_root)