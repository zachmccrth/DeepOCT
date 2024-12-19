import torch
from torch import Tensor
from torchvision.io import decode_image
from torchvision.transforms import v2
from matplotlib import pyplot as plt

transforms = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(dtype=torch.float32, scale = True),
    v2.Lambda(lambda x: x[:, 10:-10, 10:-10]),
    v2.Normalize(mean=[0], std=[1]),
    ])

img = decode_image("/home/zachary/PycharmProjects/OCTVision/resources/OCT2017/test/NORMAL/NORMAL-9251-1.jpeg")

out: Tensor = transforms(img)

plt.title("Transformed Image")
plt.imshow(out.squeeze(), cmap="gray")
plt.show()