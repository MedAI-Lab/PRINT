"""Ultrasound transforms used in PRINT training and inference."""
import torch
from PIL import Image
import torchvision.transforms as T

class PerImageNormalize:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        return (x - mean) / (std + self.eps)


def build_transforms(use_grayscale=True, mean=None, std=None, training=True):
    ops = []
    ops.append(T.Resize((224, 224)))
    
    if training:
        if use_grayscale:
            ops.append(T.Grayscale(num_output_channels=3))
        ops.append(T.RandomHorizontalFlip(p=0.5))
        ops.append(T.RandomRotation(degrees=5))
        ops.append(T.RandomAffine(degrees=0, translate=(0.02, 0.02)))
        ops.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0))
    else:
        if use_grayscale:
            ops.append(T.Grayscale(num_output_channels=3))

    ops.append(T.ToTensor())
    if training:
        ops.append(T.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'))
    ops.append(PerImageNormalize(eps=1e-6))
    return T.Compose(ops)


def load_views(paths):
    transform = build_transforms(use_grayscale=True, training=False)
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("L")))
    return torch.stack(images).unsqueeze(0)


