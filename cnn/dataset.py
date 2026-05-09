import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class BrailleCellDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        """
        Args:
            image_paths (list of str): Paths to the 64x64 cell images.
            labels (list of int, optional): Corresponding integer classes (0-63).
            transform: torchvision transforms.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Ensure grayscale

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)

        return image
