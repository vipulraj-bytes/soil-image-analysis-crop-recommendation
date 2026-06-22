import os
import cv2
import albumentations as A
from albumentations.augmentations.transforms import *
import uuid

# Define augmentation pipeline (4 augmentations per image)
augment = A.Compose([
    A.Resize(height=256, width=256),                 # Resize to standard size
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.GaussianBlur(p=0.3),
])
# Define path to train directory
base_dir = r'C:\Users\DAS\Soil_Crop_Recommendation\backend\soil_dataset\train'

# Process each soil type folder
for soil_type in os.listdir(base_dir):
    soil_path = os.path.join(base_dir, soil_type)
    if not os.path.isdir(soil_path):
        continue

    print(f"Augmenting images in: {soil_type}")
    
    images = [img for img in os.listdir(soil_path) if img.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    for img_name in images:
        img_path = os.path.join(soil_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load {img_name}")
            continue

        for i in range(4):  # create 4 augmented versions
            augmented = augment(image=image)['image']
            save_path = os.path.join(soil_path, f"{os.path.splitext(img_name)[0]}_aug{i}_{uuid.uuid4().hex[:6]}.jpg")
            cv2.imwrite(save_path, augmented)

print("✅ Augmentation complete for all soil types.")
