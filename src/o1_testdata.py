import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import torchvision.models as models
from torchmetrics.classification import Accuracy, ConfusionMatrix
import torch
from torch.utils.data import DataLoader

from src import resnet18_model_def

# ==========================
# Later (or in a new session), load the model for inference
# ==========================
# Re-initialize the same model architecture
inference_model = models.resnet18(pretrained=False)  # Pretrained=False, since we'll load our own weights
num_features = inference_model.fc.in_features
inference_model.fc = nn.Linear(num_features, 4)

# Load the saved weights
save_path = "/home/zachary/PycharmProjects/OCTVision/model_weights.pth"
inference_model.load_state_dict(torch.load(save_path))
print("Model weights loaded successfully.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_model.to(device)

# Set model to evaluation mode
inference_model.eval()

test_dataset = datasets.ImageFolder(
    root="/home/zachary/PycharmProjects/OCTVision/resources/OCT2017/test",
    transform=resnet18_model_def.preprocessing_pipeline
)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Assuming test_loader, inference_model, and device are already defined
# Initialize torchmetrics metrics
num_classes = len(test_loader.dataset.classes)  # Update this based on your dataset
accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
confusion_matrix_metric = ConfusionMatrix(num_classes=num_classes, task="multiclass").to(device)

# Switch the model to evaluation mode
inference_model.eval()

# Perform evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        # Move data to the appropriate device
        images = images.to(device)
        labels = labels.to(device)

        # Perform inference
        outputs = inference_model(images)
        _, predicted = torch.max(outputs, 1)

        # Accumulate predictions and labels for metrics
        all_preds.append(predicted)
        all_labels.append(labels)

# Concatenate all predictions and labels
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

# Calculate metrics using torchmetrics
accuracy = accuracy_metric(all_preds, all_labels)
confusion_matrix = confusion_matrix_metric(all_preds, all_labels)

# Print results
print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix.cpu().numpy())