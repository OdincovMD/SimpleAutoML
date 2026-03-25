import yaml
import os
from backend.exception.file_system import LabelError, TxtFileNotFoundError, NotEnoughImagesError
from ml.augmentation import save_with_augmentations
from ml.quiet import tqdm_disable
from ml.seed import set_seed
import random
import shutil
from tqdm import tqdm

class DataSpliting():
    def __init__(self, path_to_dataset, random_seed=42, shuffle=False):
        self.path_to_dataset = path_to_dataset
        self.random_seed = random_seed
        set_seed(self.random_seed)
        self.shuffle = shuffle
    
    @staticmethod
    def save_files_to_dir(files, image_dir, label_dir, dest_image_dir, dest_label_dir, desc):
        for file in tqdm(files, desc=desc, disable=tqdm_disable()):
            image_path = os.path.join(image_dir, file)
            dest_image_path = os.path.join(dest_image_dir, file)

            label_file = os.path.splitext(file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            dest_label_path = os.path.join(dest_label_dir, label_file)
            try:
                shutil.copy(label_path, dest_label_path)
                shutil.copy(image_path, dest_image_path)

            except Exception:
                raise TxtFileNotFoundError(label_file, label_dir)

    def spliting_seg(self, train_size=0.9, val_size=0.1, test_size=0.0, interactive=True, output_dir=None):
        if self._get_valsize(val_size, os.path.join(self.path_to_dataset, 'images')) == 0:
            raise(NotEnoughImagesError(self.path_to_dataset))

        self.output_dir = output_dir or 'data_root'
        train_image_dir = os.path.join(self.output_dir, 'train', 'images')
        train_label_dir = os.path.join(self.output_dir, 'train', 'labels')

        val_image_dir = os.path.join(self.output_dir, 'val', 'images')
        val_label_dir = os.path.join(self.output_dir, 'val', 'labels')

        test_image_dir = os.path.join(self.output_dir, 'test', 'images') if test_size > 0 else None
        test_label_dir = os.path.join(self.output_dir, 'test', 'labels') if test_size > 0 else None
        
        os.makedirs(train_image_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)

        os.makedirs(val_image_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

        if test_size > 0:
            os.makedirs(test_image_dir, exist_ok=True)
            os.makedirs(test_label_dir, exist_ok=True)

        image_dir = os.path.join(self.path_to_dataset, 'images')
        label_dir = os.path.join(self.path_to_dataset, 'labels')
        image_files = sorted(os.listdir(image_dir))
        
        if self.shuffle:
            random.shuffle(image_files)

        num_files = len(image_files)
        train_end = int(num_files * train_size)
        val_end = train_end + int(num_files * val_size)
        
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:] if test_size > 0 else []

        self.save_files_to_dir(train_files, image_dir, label_dir, train_image_dir, train_label_dir, desc="Copying train files")
        self.save_files_to_dir(val_files, image_dir, label_dir, val_image_dir, val_label_dir, desc="Copying val files")

        if test_size > 0:
            self.save_files_to_dir(test_files, image_dir, label_dir, test_image_dir, test_label_dir, desc="Copying test files")
        if interactive:
            self.building_yaml()
        else:
            self.building_yaml_auto()

    def building_yaml(self):
        unique_chars = set()
        directory = os.path.join(self.path_to_dataset, 'labels')

        if not os.path.exists(directory):
            raise FileNotFoundError(f"Директория {directory} не найдена")
        
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        for line in file:
                            line = line.strip()
                            if line:
                                unique_chars.add(line[0])
                except Exception as e:
                    raise IOError(f"Ошибка при чтении файла {filepath}: {e}")
            else:
                raise LabelError(filename)

        self.names = []
        try:
            for el in sorted(map(int, unique_chars)):
                name = input(f'Класс : {el}. Наименование: ')
                self.names.append(name)
        except ValueError:
            raise ValueError("Все уникальные символы в аннотациях должны быть числами")
        self.create_yaml(self.names, self.output_dir)

    def building_yaml_auto(self):
        unique_chars = set()
        directory = os.path.join(self.path_to_dataset, 'labels')
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Директория {directory} не найдена")
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    for line in file:
                        line = line.strip()
                        if line:
                            unique_chars.add(line[0])
            else:
                raise LabelError(filename)
        self.names = [f"class_{i}" for i in sorted(map(int, unique_chars))]
        self.create_yaml(self.names, self.output_dir)

    def create_yaml(self, names, output_folder):
        data = {
            'train': os.path.join('train', 'images'),
            'val': os.path.join('val', 'images'),
            'nc': len(names),
            'names': names
        }
        output_path = os.path.join(output_folder, 'dataset.yaml')
        try:
            with open(output_path, 'w') as file:
                yaml.dump(data, file, default_flow_style=None, allow_unicode=True)
        except Exception as e:
            raise IOError(f"Ошибка при записи YAML-файла: {e}") 
        self.output_dir = output_path
    
    @staticmethod
    def copy_files(new_train_paths, destination_folder):
        os.makedirs(destination_folder, exist_ok=True)

        for src_path in new_train_paths:
            if os.path.exists(src_path):
                file_name = os.path.basename(src_path)
                dest_path = os.path.join(destination_folder, file_name)
                shutil.copy2(src_path, dest_path)

    def spliting_class(self, train_size=0.9, val_size=0.1, output_dir=None):
        self.output_dir = output_dir or 'data_root'
        train_dir = os.path.join(self.output_dir, 'train')
        val_dir = os.path.join(self.output_dir, 'val')

        self.names = os.listdir(self.path_to_dataset)

        class_count = {}
        for class_name in self.names:
            class_dir = os.path.join(self.path_to_dataset, class_name)
            class_count[class_name] = len(os.listdir(class_dir))

        max_class_name, max_class_count = max(class_count.items(), key=lambda item: item[1])

        for class_name in self.names:
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

            source_dir = os.path.join(self.path_to_dataset, class_name)
            class_files = os.listdir(source_dir)
            
            if self.shuffle:
                random.shuffle(class_files)

            if self._get_valsize(val_size, source_dir) == 0:
                raise(NotEnoughImagesError(self.path_to_dataset))

            num_files = len(class_files)
            train_end = int(num_files * train_size)
            val_end = train_end + int(num_files * val_size)
            
            train_files = class_files[:train_end]
            val_files = class_files[train_end:val_end]

            if class_name != max_class_name:
                augment_factor = max_class_count // class_count[class_name]
            else:
                augment_factor = 0
            save_with_augmentations(train_files, source_dir, train_dir, class_name, desc=f"dir: train | class: {class_name}", augment_factor=augment_factor)
            save_with_augmentations(val_files, source_dir, val_dir, class_name, desc=f"dir: val | class: {class_name}")

    def _get_valsize(self, val_size, image_dir):
        image_files = os.listdir(image_dir)
        num_files = len(image_files)
        valsize = int(num_files * val_size)
        return valsize
