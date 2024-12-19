import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import torchvision.models as models

# Define your transformations
# Note: We'll replicate the single-channel grayscale image three times
# to be compatible with a pretrained ResNet (which expects 3-channel input).
train_transforms = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.Resize((224, 224)),  # Resize all images to 224x224 (size used for ResNet models)
    v2.ToTensor(),  # Convert PIL.Image to torch.Tensor
    v2.Lambda(lambda x: x[:, 10:-10, 10:-10]),
    v2.Lambda(lambda x: x.repeat(3, 1, 1)),
    v2.Normalize(mean=[0], std=[1])
])

# Create the dataset using ImageFolder
train_dataset = datasets.ImageFolder(
    root='/home/zachary/PycharmProjects/OCTVision/resources/OCT2017/train',
    transform=train_transforms
)

# Create the DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# Load a pretrained ResNet model
model = models.resnet18(weights='DEFAULT')

# If you want to fine-tune the model for your 4-class problem:
# Replace the last fully connected layer with a new one that has 4 outputs
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
