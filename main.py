import dataset.load_dataset as load_dataset

load_type = input('Тип загрузки (drive/zip): ')

folder  = load_dataset.main(load_type)