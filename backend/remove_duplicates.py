import os
import hashlib

def get_image_hash(image_path):
    """Calculate hash of an image using SHA256"""
    with open(image_path, "rb") as f:
        img_hash = hashlib.sha256(f.read()).hexdigest()
    return img_hash

def remove_duplicates(image_folder):
    """Remove duplicate images from a folder"""
    seen_hashes = set()
    duplicates = []

    # Loop through all the files in the folder
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif', 'tif', 'webp')):
                image_path = os.path.join(root, file)
                image_hash = get_image_hash(image_path)
                
                # If this hash is already in seen_hashes, it's a duplicate
                if image_hash in seen_hashes:
                    duplicates.append(image_path)  # Keep track of duplicates
                else:
                    seen_hashes.add(image_hash)

    # Remove duplicates
    for duplicate in duplicates:
        os.remove(duplicate)
        print(f"Deleted duplicate image: {duplicate}")

# Specify the path to your train dataset
train_image_folder = 'C:/Users/DAS/Soil_Crop_Recommendation/backend/soil_dataset/train/'

remove_duplicates(train_image_folder)
