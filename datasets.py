# datasets.py

import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple


class MnistDataset(Dataset):
    """
    Custom Dataset for loading MNIST data from CSV or image files.

    Args:
        path (str): Path to the CSV file or image directory.
        is_csv (bool, optional): Flag indicating if the path is a CSV file.
            Defaults to True.
    """
    def __init__(self, path: str, is_csv: bool = True):
        self.images, self.labels = self._load_data(path=path, is_csv=is_csv)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]

    def _load_data(
        self,
        path: str,
        is_csv: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if is_csv:
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV file not found at {path}")
            try:
                data = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                raise ValueError(f"CSV file at {path} is empty or corrupted.")
            # Load data from CSV file
            data = pd.read_csv(path)
            labels = torch.tensor(data.iloc[:, 0].values, dtype=torch.long)
            images = torch.tensor(
                data.iloc[:, 1:].values, dtype=torch.float32
            ).reshape(-1, 1, 28, 28)
        else:
            if not os.path.isdir(path):
                raise NotADirectoryError(f"Directory not found at {path}")
            # Load data from .png files
            images = []
            labels = []
            for image_file in os.listdir(path):
                if image_file.upper().endswith(".PNG"):
                    img_path = os.path.join(path, image_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (28, 28))
                    img = cv2.bitwise_not(img)  # Invert colors
                    images.append(img)
                    labels.append(int(image_file.split("_")[0]))
            images = torch.tensor(
                np.array(images), dtype=torch.float32
            ).reshape(-1, 1, 28, 28)
            labels = torch.tensor(labels, dtype=torch.long)

        # Normalize images
        transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        images = transform(images)

        return images, labels
