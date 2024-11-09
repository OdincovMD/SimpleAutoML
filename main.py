import os
import dataset.load_dataset as load_dataset
from dataset.preprocessing import train_val_split, create_yaml
from src.queries.orm import SyncOrm 

load_type = input('Тип загрузки (drive/zip): ')
folder = load_dataset.main(load_type)

for root, dirs, files in os.walk(folder):
    if os.path.basename(root) not in ('test', 'labels'):
        for file in files:
            SyncOrm.insert_data({'train_folder': folder, 'path': os.path.join(root, file)})

# Проверка, существует ли модель для текущей папки
if not SyncOrm.select_model(folder):
    dataset_path = os.path.join(folder, 'dataset')

    path_to_folder_train_val =  train_val_split(dataset_path) 

    create_yaml(folder, path_to_folder_train_val)
    
    # model = Обучение(dataset_path) # Здесь происходит обучение модели
    model_path = 'model_path'
    SyncOrm.update_data(folder)
    SyncOrm.insert_model({'train_folder': folder, 'path': model_path})

# # Если есть данные для дообучения, дообучаем модель
# elif add_training := SyncOrm.select_data(folder):
#     dataset_path = os.path.join(download_path, 'dataset')
#     new_train_paths = [os.path.join(dataset_path, entry[0]) for entry in add_training]
#     # model = Дообучение(new_train_paths) # Здесь происходит дообучение модели
#     model_path = 'model_path'
#     SyncOrm.update_data(folder)
#     SyncOrm.update_model(folder, model_path)

# # Если модель уже существует, проводим инференс
# else:
#     model = SyncOrm.select_model(folder)
#     test_path = os.path.join(download_path, 'test')
#     for filename in os.listdir(test_path):
#         result = model(os.path.join(test_path, filename))
#         print(result)