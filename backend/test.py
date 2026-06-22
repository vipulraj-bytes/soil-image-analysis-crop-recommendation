import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load class names
class_names = sorted(os.listdir("soil_dataset/train"))

# Load the model
model = torch.load("soil_model.pkl", map_location=device, weights_only=False)
model.eval()

# Choose a test image
test_image_path = "soil_dataset/test/Laterite/Lateric-soil-(1).png"  # 🔁 replace with your actual test image path

# Load and transform the image
image = Image.open(test_image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# Perform prediction
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

# Show result
print(f"Predicted class: {class_names[predicted.item()]}")
