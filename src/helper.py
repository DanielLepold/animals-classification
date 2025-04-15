import shutil
import os
from sklearn.model_selection import train_test_split

data_dir = "./raw-img"
classes = os.listdir(data_dir)


os.makedirs('./input/train', exist_ok=True)
os.makedirs('./input/test', exist_ok=True)

for class_name in classes:
    os.makedirs(f'./input/train/{class_name}', exist_ok=True)
    os.makedirs(f'./input/test/{class_name}', exist_ok=True)

    images = os.listdir(f"{data_dir}/{class_name}")
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

    for image in train_images:
        shutil.move(f"{data_dir}/{class_name}/{image}", f"./input/train/{class_name}")

    for image in test_images:
        shutil.move(f"{data_dir}/{class_name}/{image}", f"./input/test/{class_name}")
