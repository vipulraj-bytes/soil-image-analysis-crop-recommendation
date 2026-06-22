import os
from PIL import Image

data_dir = "./soil_dataset"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

corrupted_files = []

for root, _, files in os.walk(train_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verify image integrity
            except Exception as e:
                print(f"Corrupted train file: {file_path} - {e}")
                corrupted_files.append(file_path)

for root, _, files in os.walk(test_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()
            except Exception as e:
                print(f"Corrupted test file: {file_path} - {e}")
                corrupted_files.append(file_path)

print(f"Total corrupted files found: {len(corrupted_files)}")