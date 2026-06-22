# backend/train_cnn.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm # Import tqdm

# --- Configuration ---
# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"INFO: Using device: {device}")

# Dataset paths (relative to this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "soil_dataset") # Expecting soil_dataset folder in backend/
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Check if dataset directories exist
if not os.path.isdir(train_dir):
    raise FileNotFoundError(f"ERROR: Training directory not found at {train_dir}")
if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"ERROR: Testing directory not found at {test_dir}")

# Image parameters
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
NUM_WORKERS = 0 # Set to 0 for Windows compatibility or > 0 for parallel loading on Linux/Mac if needed

# Transformations
# Train transforms include augmentation
train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet stats
])

# Validation/Test transforms - only resize, ToTensor, Normalize
val_test_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Load datasets ---
print("INFO: Loading datasets...")
# Load full training data first to get labels for stratification
try:
    # Use val_test_transforms here just for initial loading/splitting consistency
    full_train_dataset_for_split = datasets.ImageFolder(train_dir, transform=val_test_transforms)
    if not full_train_dataset_for_split.classes:
         raise ValueError("No classes found in the training directory.")
    NUM_CLASSES = len(full_train_dataset_for_split.classes)
    print(f"INFO: Found {NUM_CLASSES} classes: {full_train_dataset_for_split.classes}")
except Exception as e:
    print(f"ERROR: Failed to load initial training dataset for splitting: {e}")
    raise SystemExit(e)

# Split train into train and validation indices
try:
    train_idx, val_idx = train_test_split(
        list(range(len(full_train_dataset_for_split))),
        test_size=0.2, # 20% for validation
        random_state=42, # for reproducibility
        stratify=full_train_dataset_for_split.targets # Ensure class distribution is similar in splits
    )
    print(f"INFO: Splitting training data: {len(train_idx)} train, {len(val_idx)} validation samples.")
except Exception as e:
    print(f"ERROR: Failed to split training data: {e}")
    raise SystemExit(e)

# Create actual datasets with appropriate transforms
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms) # Apply train transforms
val_dataset = datasets.ImageFolder(train_dir, transform=val_test_transforms)   # Apply val transforms

# Apply the indices splits using Subset
train_dataset = Subset(train_dataset, train_idx)
val_dataset = Subset(val_dataset, val_idx)

# Create test dataset
try:
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)
    if len(test_dataset) == 0:
         print("WARNING: Test dataset is empty.")
except Exception as e:
    print(f"ERROR: Failed to load test dataset: {e}")
    raise SystemExit(e)


# --- DataLoaders ---
print("INFO: Creating DataLoaders...")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True if device=='cuda' else False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if device=='cuda' else False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if device=='cuda' else False)
print("INFO: DataLoaders created.")


# Debug print: Show dataset sizes and one batch shape
print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
try:
    # Check if train_loader is not empty
    if len(train_loader) > 0:
        for images, labels in train_loader:
            print(f"DEBUG: Sample batch loaded - Images shape: {images.shape}, Labels shape: {labels.shape}")
            break # Only need to see one batch
    else:
        print("WARNING: Train loader is empty!")
except Exception as e:
     print(f"ERROR: Could not load a batch from train_loader: {e}")
# --- End Data Setup ---


# --- Custom CNN Model Definition ---
class SoilCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES): # Use NUM_CLASSES found earlier
        super(SoilCNN, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Dropout for regularization

        # Calculate flattened size dynamically (assuming input 224x224)
        # After 3 pooling layers (224 -> 112 -> 56 -> 28)
        self.flattened_size = 128 * 28 * 28

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512) # Increased hidden size
        self.fc2 = nn.Linear(512, num_classes)


    def forward(self, x):
        # Conv blocks: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # Flatten all dimensions except batch

        # Fully connected layers with dropout
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x) # Output layer (logits)
        return x
# --- End Model Definition ---


# --- Training Setup ---
print("INFO: Initializing model, loss function, and optimizer...")
model = SoilCNN().to(device)

# Loss function (CrossEntropyLoss includes Softmax)
criterion = nn.CrossEntropyLoss()

# Optimizer (Adam is a common choice)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Variables to track best model
best_val_acc = 0.0
best_model_path = os.path.join(script_dir, "best_soil_cnn_model.pkl")
final_model_path = os.path.join(script_dir, "soil_cnn_model.pkl")
print(f"INFO: Model initialized. Ready for training on {device}.")
# --- End Training Setup ---


# === Main Training Loop ===
# Ensure this block runs only when the script is executed directly
if __name__ == '__main__':
    print(f"\n--- Starting Training for {EPOCHS} Epochs ---")
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train() # Set model to training mode
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Wrap train_loader with tqdm for a progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)

        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            batch_loss = loss.item()
            running_train_loss += batch_loss
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar description with current batch loss/accuracy
            train_pbar.set_postfix({"Loss": f"{batch_loss:.4f}", "Acc": f"{100*train_correct/train_total:.2f}%"})

        avg_train_loss = running_train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_pbar.close() # Close the training progress bar

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Wrap val_loader with tqdm
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)

        with torch.no_grad(): # No need to calculate gradients during validation
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_loss = loss.item()
                running_val_loss += batch_loss
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Update validation progress bar
                val_pbar.set_postfix({"Loss": f"{batch_loss:.4f}", "Acc": f"{100*val_correct/val_total:.2f}%"})

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_pbar.close() # Close the validation progress bar

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save the model if validation accuracy has improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            try:
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ Best model saved! (Epoch {epoch+1}, Val Acc: {best_val_acc:.2f}%) Path: {best_model_path}")
            except Exception as e:
                 print(f"ERROR: Could not save best model: {e}")

    print("\n--- Training Finished ---")

    # --- Testing Phase ---
    print("\n--- Starting Testing Phase ---")
    # Load the best performing model for testing
    try:
        if os.path.exists(best_model_path):
             print(f"INFO: Loading best model for testing from {best_model_path}")
             model.load_state_dict(torch.load(best_model_path, map_location=device))
        else:
             print("WARNING: Best model file not found. Testing with the final state model.")
    except Exception as e:
        print(f"ERROR: Could not load best model state dict: {e}. Testing with final state model.")

    model.eval() # Set model to evaluation mode
    test_correct = 0
    test_total = 0

    # Wrap test_loader with tqdm
    test_pbar = tqdm(test_loader, desc="[Test]", leave=False)

    with torch.no_grad():
        for images, labels in test_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            # Update test progress bar
            test_pbar.set_postfix({"Acc": f"{100*test_correct/test_total:.2f}%"})

    test_acc = 100 * test_correct / test_total
    test_pbar.close()
    print(f"\n✅ Final Test Accuracy: {test_acc:.2f}%")

    # --- Save the Final Model ---
    # Save the model state after all epochs (might not be the best performing one)
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"✅ Final model state saved as {final_model_path}")
    except Exception as e:
        print(f"ERROR: Could not save final model: {e}")

    print("\n--- Script Finished ---")
# === End Main Execution Block ===