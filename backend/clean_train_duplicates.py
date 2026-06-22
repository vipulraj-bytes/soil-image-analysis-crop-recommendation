import os
import hashlib

def remove_duplicate_images(folder_path):
    print(f"\n🧹 Checking folder: {folder_path}")
    hashes = set()
    duplicates = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)

            # Only check supported image formats
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')):
                continue

            with open(filepath, 'rb') as file:
                file_hash = hashlib.md5(file.read()).hexdigest()

            if file_hash in hashes:
                duplicates.append(filepath)
            else:
                hashes.add(file_hash)

    for dup in duplicates:
        os.remove(dup)
        print(f"🗑️ Removed duplicate: {dup}")

    print(f"\n✅ Done. Removed {len(duplicates)} duplicate images from train folder.\n")


# 👇 Only set this one line
train_path = r"C:\Users\DAS\Soil_Crop_Recommendation\backend\soil_dataset\train"
remove_duplicate_images(train_path)
