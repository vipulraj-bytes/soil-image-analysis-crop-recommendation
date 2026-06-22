import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage

# Define paths
test_data_path = "soil_dataset/test"
output_path = "soil_dataset/augmented_test"
os.makedirs(output_path, exist_ok=True)

# Define augmentation transformations
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()  # Convert to tensor first
])

# Convert tensor back to PIL
to_pil = ToPILImage()

def augment_test_data(class_name):
    input_dir = os.path.join(test_data_path, class_name)
    output_dir = os.path.join(output_path, class_name)
    os.makedirs(output_dir, exist_ok=True)

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    current_count = len(images)
    target_count = 200

    if current_count == 0:
        print(f"Skipping {class_name} (no images).")
        return

    images_needed = target_count - current_count
    if images_needed <= 0:
        print(f"{class_name} already has {current_count} or more images.")
        return

    augmentations_per_image = (images_needed // current_count) + 1

    img_index = 1
    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Save original image
        image.save(os.path.join(output_dir, f"{class_name}_{img_index}.jpg"))
        img_index += 1

        for _ in range(augmentations_per_image):
            augmented_tensor = augmentation_transforms(image)
            augmented_image = to_pil(augmented_tensor)
            augmented_image.save(os.path.join(output_dir, f"{class_name}_{img_index}.jpg"))
            img_index += 1

            if img_index > target_count:
                break
        if img_index > target_count:
            break

    print(f"{class_name} augmented to {img_index - 1} images.")

def augment_all_test_data():
    for class_name in os.listdir(test_data_path):
        augment_test_data(class_name)

if __name__ == "__main__":
    augment_all_test_data()
    print("✅ Augmentation of test data completed.")
