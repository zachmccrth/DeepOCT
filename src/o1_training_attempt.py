import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.models as models
from src import resnet18_model_def

# Define your transformations
# Note: We'll replicate the single-channel grayscale image three times
# to be compatible with a pretrained ResNet (which expects 3-channel input).

# Create the dataset using ImageFolder
train_dataset = datasets.ImageFolder(
    root='/home/zachary/PycharmProjects/OCTVision/resources/OCT2017/train',
    transform=resnet18_model_def.preprocessing_pipeline
)

# Create the DataLoader
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=6)

# Load a pretrained ResNet model
model = models.resnet18(weights='DEFAULT')

# If you want to fine-tune the model for your 4-class problem:
# Replace the last fully connected layer with a new one that has 4 outputs
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, resnet18_model_def.OUTPUT_FEATURES)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(">>> Using device:", device, "\n\n")
model.to(device)

# Example training loop snippet
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(2):  # Just an example
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} complete.")

# Assuming you have trained model (model), optimizer, etc. as before

# ==========================
# Save the trained model
# ==========================
save_path = "/home/zachary/PycharmProjects/OCTVision/model_weights.pth"
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")
