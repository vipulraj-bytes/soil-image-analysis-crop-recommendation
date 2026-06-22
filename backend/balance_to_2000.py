import os
import random
import shutil

def balance_classes(folder_path, target_count=1000):
    for class_name in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_folder):
            continue

        images = [
            f for f in os.listdir(class_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'))
        ]

        current_count = len(images)
        print(f"{class_name}: {current_count} images before balancing")

        if current_count > target_count:
            # Too many → randomly remove extras
            to_remove = random.sample(images, current_count - target_count)
            for img in to_remove:
                os.remove(os.path.join(class_folder, img))
            print(f"✂️ Trimmed {class_name} from {current_count} to {target_count} images.")

        elif current_count < target_count:
            # Too few → randomly copy images to reach target
            to_add = target_count - current_count
            to_duplicate = random.choices(images, k=to_add)
            for i, img in enumerate(to_duplicate):
                src = os.path.join(class_folder, img)
                dst = os.path.join(class_folder, f"dup_{i}_{img}")
                shutil.copyfile(src, dst)
            print(f"➕ Augmented {class_name} from {current_count} to {target_count} images.")

        else:
            print(f"✅ {class_name} already has exactly {target_count} images.")

train_path = r"C:\Users\DAS\Soil_Crop_Recommendation\backend\soil_dataset\train"
balance_classes(train_path, target_count=1000)
