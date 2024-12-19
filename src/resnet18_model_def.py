import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import torchvision.models as models

preprocessing_pipeline = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.Resize((224, 224)),  # Resize all images to 224x224 (size used for ResNet models)
    v2.ToTensor(),  # Convert PIL.Image to torch.Tensor
    v2.Lambda(lambda x: x[:, 10:-10, 10:-10]),
    v2.Lambda(lambda x: x.repeat(3, 1, 1)),
    v2.Normalize(mean=[0], std=[1])
])