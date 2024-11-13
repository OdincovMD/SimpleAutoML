import kagglehub
import os
import shutil
from sklearn.model_selection import train_test_split

path_root = r'D:\Mine\Basics\Coding\Python_projects\SimpleAutoML\ml\datasets'
os.makedirs(path_root, exist_ok=True)

# Download latest butterflies
path = kagglehub.dataset_download("deepakat002/yolov8-segmentation-data-butterfly")
print("Path to dataset files:", path)

path_root_butterfly = os.path.join(path_root, 'Butterfly')
data_root_butterfly = os.path.join(path_root_butterfly, os.listdir(path)[0])

if 'Butterfly' in os.listdir(path_root):
    shutil.rmtree(path_root_butterfly)
else:
    os.makedirs(path_root_butterfly, exist_ok=True)

shutil.copytree(path, path_root_butterfly, dirs_exist_ok=True)

for purp in ['train', 'val']:
    os.makedirs(os.path.join(path_root_butterfly, purp), exist_ok=True)
    for dir in ['images', 'labels']:
        os.makedirs(os.path.join(path_root_butterfly, purp, dir), exist_ok=True)

X_data = []
y_data = []
for obj in os.listdir(data_root_butterfly):
    image_extentions = ['png', 'jpeg', 'jpg']
    if obj.partition('.')[2] in image_extentions:
        X_data.append(obj)
    elif obj.endswith(".txt"):
        y_data.append(obj)
        obj = os.path.join(data_root_butterfly, obj)
        content = []
        with open(obj, "rt", encoding="utf-8") as read:
            for line in read.readlines():
                new_line = line.split(" ")
                new_line[0] = str(0)
                content.append(" ".join(new_line))
        with open(obj, "wt", encoding="utf-8") as write:
            write.writelines(content)

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, train_size=0.7, shuffle=True, random_state=42)

for X, y, purp in ((X_train, y_train, 'train'), (X_val, y_val, 'val')):
    for file, label in zip(X, y):
        shutil.move(os.path.join(data_root_butterfly, file), os.path.join(path_root_butterfly, purp, 'images', file))
        shutil.move(os.path.join(data_root_butterfly, label), os.path.join(path_root_butterfly, purp, 'labels', label))

shutil.rmtree(data_root_butterfly)

with open(os.path.join(path_root_butterfly, 'data.yaml'), 'wt', encoding="utf-8") as conf:
    conf.write(f"train: ../train/images\n")
    conf.write(f"val: ../val/images\n")
    conf.write(f"\n")
    conf.write(f"nc: 1\n")
    conf.write(f"names: ['Butterfly']\n")

# Download the grapes
path = kagglehub.dataset_download("nicolaasregnier/grape-grapes")
print("Path to dataset files:", path)

path_root_grape = os.path.join(path_root, 'Grape')
data_root_grape = os.path.join(path_root_grape, os.listdir(path)[os.listdir(path).index("SAMPreds")])

if 'Grape' in os.listdir(path_root):
    shutil.rmtree(path_root_grape)
else:
    os.makedirs(path_root_grape, exist_ok=True)

shutil.copytree(path, path_root_grape, dirs_exist_ok=True)
os.remove(os.path.join(path_root_grape, "data.yaml"))
for dir in os.listdir(path_root_grape):
    if dir != 'SAMPreds':
        shutil.rmtree(os.path.join(path_root_grape, dir))

for purp in ['train', 'val']:
    os.makedirs(os.path.join(path_root_grape, purp), exist_ok=True)
    for dir in ['images', 'labels']:
        os.makedirs(os.path.join(path_root_grape, purp, dir), exist_ok=True)

X_data = os.listdir(os.path.join(data_root_grape, 'images'))
y_data = os.listdir(os.path.join(data_root_grape, 'labels'))

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, train_size=0.7, shuffle=True, random_state=42)

for X, y, purp in ((X_train, y_train, 'train'), (X_val, y_val, 'val')):
    for file, label in zip(X, y):
        shutil.move(os.path.join(data_root_grape, 'images', file), os.path.join(path_root_grape, purp, 'images', file))
        shutil.move(os.path.join(data_root_grape, 'labels', label), os.path.join(path_root_grape, purp, 'labels', label))

shutil.rmtree(data_root_grape)

with open(os.path.join(path_root_grape, 'data.yaml'), 'wt', encoding="utf-8") as conf:
    conf.write(f"train: ../train/images\n")
    conf.write(f"val: ../val/images\n")
    conf.write(f"\n")
    conf.write(f"nc: 4\n")
    conf.write(f"names: ['Chardonnay', 'PinotNoir', 'PinotGris', 'SauvignonBlanc']\n")

# Download the potholes
path = kagglehub.dataset_download("farzadnekouei/pothole-image-segmentation-dataset")
print("Path to dataset files:", path)

path_root_pothole = os.path.join(path_root, 'Pothole')
data_root_pothole = os.path.join(path_root_pothole, os.listdir(path)[0])

if 'Pothole' in os.listdir(path_root):
    shutil.rmtree(path_root_pothole)
else:
    os.makedirs(path_root_pothole, exist_ok=True)

shutil.copytree(path, path_root_pothole, dirs_exist_ok=True)

for dir in ('images', 'labels'):
    os.makedirs(os.path.join(path_root_pothole, dir), exist_ok=True)

for purp in ['train', 'val']:
    os.makedirs(os.path.join(path_root_pothole, purp), exist_ok=True)
    for dir in ['images', 'labels']:
        os.makedirs(os.path.join(path_root_pothole, purp, dir), exist_ok=True)

for purp in ('train', 'valid'):
    for src in ('images', 'labels'):
        for file in os.listdir(os.path.join(data_root_pothole, purp, src)):
            shutil.move(os.path.join(data_root_pothole, purp, src, file), os.path.join(path_root_pothole, src))

shutil.rmtree(data_root_pothole)

X_data = os.listdir(os.path.join(path_root_pothole, 'images'))
y_data = os.listdir(os.path.join(path_root_pothole, 'labels'))

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, train_size=0.7, shuffle=True, random_state=42)

for X, y, purp in ((X_train, y_train, 'train'), (X_val, y_val, 'val')):
    for file, label in zip(X, y):
        shutil.move(os.path.join(path_root_pothole, 'images', file), os.path.join(path_root_pothole, purp, 'images', file))
        shutil.move(os.path.join(path_root_pothole, 'labels', label), os.path.join(path_root_pothole, purp, 'labels', label))

for dir in ('images', 'labels'):
    shutil.rmtree(os.path.join(path_root_pothole, dir))

with open(os.path.join(path_root_pothole, 'data.yaml'), 'wt', encoding="utf-8") as conf:
    conf.write(f"train: ../train/images\n")
    conf.write(f"val: ../val/images\n")
    conf.write(f"\n")
    conf.write(f"nc: 1\n")
    conf.write(f"names: ['Pothole']\n")