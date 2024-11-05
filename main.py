import os
import dataset.load_dataset as load_dataset
from src.queries.orm import SyncOrm 

load_type = input('Тип загрузки (drive/zip): ')

folder  = load_dataset.main(load_type)
# SyncOrm.create_tables()

for root, dirs, files in os.walk(f'downloads/{folder}'):
    for file in files:
        SyncOrm.insert_data({'train_folder': folder, 'path': file})

if not SyncOrm.select_model(folder):
    #model = Обучение()
    model_path = 'model_path'
    SyncOrm.update_data(folder)
    SyncOrm.insert_model({'train_folder': folder, 'path': model_path})
elif add_training := SyncOrm.select_data(folder):
    #model = Дообучнение()
    model_path = 'model_path'
    SyncOrm.update_data(folder)
    SyncOrm.update_model(folder, model_path)
else:
    pass
    #model = Тест()