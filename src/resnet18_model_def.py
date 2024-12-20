from typing import Callable, Any

import PIL
import numpy as np

INPUT_FEATURES = 224
OUTPUT_FEATURES = 4

import albumentations as A
from albumentations.pytorch import ToTensorV2


def albumentations_to_pytorch_transform(transform: Callable) -> Callable:
    def wrapper(image, **kwargs):
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)
        return transform(image=image, **kwargs)['image']
    return wrapper

preprocessing_pipeline = albumentations_to_pytorch_transform(
    A.Compose([
        A.ToGray(always_apply=True),  # Converts the image to grayscale
        A.Resize(INPUT_FEATURES, INPUT_FEATURES),  # Resize all images to the desired size
        A.Normalize(mean=[0.4], std=[0.2]),  # Normalize the image based on the given mean & std
        ToTensorV2()  # Converts the numpy array to PyTorch tensor, ensuring dtype float32
    ]))

