# models.py

import torch
from torch import nn
from abc import ABC, abstractmethod


class MnistBase(ABC, nn.Module):
    """
    Abstract base class for MNIST models.

    Args:
        dropout_rate (float, optional): Dropout rate for the model.
            Defaults to 0.3.
    """
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dropout_rate = dropout_rate
        self.layers = self._create_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @abstractmethod
    def _create_layers(self) -> nn.Sequential:
        pass


class MnistLinear(MnistBase):
    """
    A fully connected neural network for MNIST classification.

    Args:
        dropout_rate (float, optional): Dropout rate for the model.
            Defaults to 0.3.
    """
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__(dropout_rate)

    def _create_layers(self) -> nn.Sequential:
        return nn.Sequential(
            self.flatten,
            nn.Linear(28 * 28, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )


class MnistConv(MnistBase):
    """
    A convolutional neural network for MNIST classification.

    Args:
        dropout_rate (float, optional): Dropout rate for the model.
            Defaults to 0.3.
    """
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__(dropout_rate)

    def _create_layers(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_rate),
            self.flatten,
            nn.Linear(9216, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )


class MnistResNet(MnistBase):
    """
    A residual neural network for MNIST classification.

    Args:
        dropout_rate (float, optional): Dropout rate for the model. Defaults
            to 0.3.
    """
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__(dropout_rate)

    def _create_layers(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
