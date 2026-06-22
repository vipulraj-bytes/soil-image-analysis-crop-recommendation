import os
from PIL import Image

data_dir = "./soil_dataset"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

for root, _, files in os.walk(train_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(os.path.join(root, file))
                img.verify()
            except Exception as e:
                print(f"Corrupted train file: {os.path.join(root, file)} - {e}")

for root, _, files in os.walk(test_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(os.path.join(test_dir, file))
                img.verify()
            except Exception as e:
                print(f"Corrupted test file: {os.path.join(root, file)} - {e}")