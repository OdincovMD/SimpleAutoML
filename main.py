import os
import shutil
import dataset.load_dataset as load_dataset
from dataset.preprocessing import train_val_test_split, create_yaml, building_yaml, copy_files
from src.queries.orm import SyncOrm 

# SyncOrm.create_tables()
load_type = input('Тип загрузки (drive/zip): ')
folder = load_dataset.main(load_type)
dataset_path = os.path.join(folder, 'dataset')

for root, dirs, files in os.walk(folder):
    if os.path.basename(root) not in ('test', 'labels'):
        for file in files:
            SyncOrm.insert_data({'train_folder': folder, 'path': os.path.join(root, file)})

# Проверка, существует ли модель для текущей папки
if not SyncOrm.select_model(folder):
    data_root =  train_val_test_split(dataset_path) 
    label = building_yaml(dataset_path, data_root)
    # model = Обучение(dataset_path) # Здесь происходит обучение модели
    model_path = 'model_path'
    SyncOrm.update_data(folder)
    SyncOrm.insert_model({'train_folder': folder, 'path': model_path, 'classes': label})
    shutil.rmtree(f'{data_root}')

# Если есть данные для дообучения, дообучаем модель
elif add_training := SyncOrm.select_data(folder):
    data_root = './data_root'
    model_path, classes = SyncOrm.select_model(folder)[0]
    new_train_paths = [os.path.join(dataset_path, entry[0]) for entry in add_training]
    copy_files(new_train_paths, destination_folder=data_root)
    create_yaml(classes, data_root)
    # model = Дообучение(new_train_paths) # Здесь происходит дообучение модели
    model_path = 'model_path'
    SyncOrm.update_data(folder)
    SyncOrm.update_model(folder, model_path)
    shutil.rmtree(f'{data_root}')

# Если модель уже существует, проводим инференс
else:
    model = SyncOrm.select_model(folder)[0][0]
    test_path = os.path.join(folder, 'test')
    for filename in os.listdir(test_path):
        result = model(os.path.join(test_path, filename))
        print(result)