from torchvision.transforms import v2

INPUT_FEATURES = 224
OUTPUT_FEATURES = 4

preprocessing_pipeline = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.Resize((INPUT_FEATURES, INPUT_FEATURES)),  # Resize all images to 224x224 (size used for ResNet models)
    v2.ToTensor(),  # Convert PIL.Image to torch.Tensor
    v2.Lambda(lambda x: x[:, 10:-10, 10:-10]),
    v2.Lambda(lambda x: x.repeat(3, 1, 1)),
    v2.Normalize(mean=[0], std=[1])
])

