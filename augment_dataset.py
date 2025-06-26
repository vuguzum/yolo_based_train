import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path

# Путь к исходным данным
IMAGE_PATH = "project/images/train"
LABEL_PATH = "project/labels/train"
AUG_IMAGE_PATH = "project/augmented/images"
AUG_LABEL_PATH = "project/augmented/labels"

# Количество аугментированных версий на одно изображение
AUG_COUNT = 5

# Создание папок для аугментированных данных
os.makedirs(AUG_IMAGE_PATH, exist_ok=True)
os.makedirs(AUG_LABEL_PATH, exist_ok=True)

# Функция чтения аннотаций
def read_labels(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    bboxes = []
    class_labels = []
    for line in lines:
        data = list(map(float, line.strip().split()))
        class_id = int(data[0])
        bbox = data[1:]  # x_center, y_center, width, height
        bboxes.append(bbox)
        class_labels.append(class_id)
    return bboxes, class_labels

# Функция сохранения аннотаций
def save_labels(save_path, bboxes, class_labels):
    with open(save_path, 'w') as file:
        for cls, bbox in zip(class_labels, bboxes):
            line = f"{cls} {' '.join(map(str, bbox))}\n"
            file.write(line)

# Определение аугментаций
def get_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Blur(blur_limit=3, p=0.2),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),

        # Один из вариантов ниже:
        A.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        # или
        # A.RandomResizedCropLegacy(height=640, width=640, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Основной процесс аугментации
transform = get_transform()

image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith(('.jpg', '.png', '.jpeg'))]

for img_file in image_files:
    image = cv2.imread(os.path.join(IMAGE_PATH, img_file))
    image_h, image_w = image.shape[:2]

    label_file = img_file.replace(Path(img_file).suffix, ".txt")
    bboxes, class_labels = read_labels(os.path.join(LABEL_PATH, label_file))

    for i in range(AUG_COUNT):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_class_labels = augmented['class_labels']

        # Сохранение изображения
        aug_img_name = f"{Path(img_file).stem}_aug{i}{Path(img_file).suffix}"
        cv2.imwrite(os.path.join(AUG_IMAGE_PATH, aug_img_name), aug_image)

        # Сохранение аннотаций
        aug_label_name = f"{Path(label_file).stem}_aug{i}.txt"
        save_labels(os.path.join(AUG_LABEL_PATH, aug_label_name), aug_bboxes, aug_class_labels)

print("✅ Аугментация завершена!")