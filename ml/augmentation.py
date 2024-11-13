import os
import shutil
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
])

def save_with_augmentations(files, source_dir, dest_dir, class_name, desc, augment_factor=0):
    """
    Сохранение файлов с аугментацией изображений для увеличения количества примеров класса.

    Параметры:
        files (list): Список имен файлов для обработки.
        source_dir (str): Путь к исходной директории с изображениями.
        dest_dir (str): Путь к директории для сохранения обработанных изображений.
        class_name (str): Имя класса, используемое для создания поддиректории в `dest_dir`.
        desc (str): Описание для прогресс-бара tqdm.
        augment_factor (int): Количество раз, которое нужно применить аугментацию к каждому изображению.
    Возвращает:
        None. Функция сохраняет изображения с аугментацией в указанной директории.
    """
    os.makedirs(os.path.join(dest_dir, class_name), exist_ok=True)
    
    for file in tqdm(files, desc=desc):
        original_image_path = os.path.join(source_dir, file)
        save_path = os.path.join(dest_dir, class_name, file)
            
        shutil.copy(original_image_path, save_path)

        if augment_factor:
            image = Image.open(original_image_path).convert("RGB")
            for i in range(augment_factor):
                augmented_image = augment_transform(image)
                augmented_save_path = os.path.join(dest_dir, class_name, f"{i}_{file}")
                augmented_image.save(augmented_save_path)