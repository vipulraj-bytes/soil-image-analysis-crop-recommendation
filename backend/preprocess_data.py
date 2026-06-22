import os
import cv2
import numpy as np
import pickle

# Define your labels
labels = {"alluvial": 0, "black": 1, "clay": 2, "red": 3}

X, y = [], []

# Load and process images
for category in labels.keys():
    folder_path = os.path.join("soil_dataset/train", category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))  # Resize image to 128x128
        X.append(img)
        y.append(labels[category])

# Convert to NumPy array and normalize
X = np.array(X) / 255.0  
y = np.array(y)

# Save preprocessed data for training
with open("processed_data.pkl", "wb") as file:
    pickle.dump((X, y), file)

print("✅ Data Preprocessing Completed! Processed data saved as processed_data.pkl")
