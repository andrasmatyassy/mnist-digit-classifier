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
            # Flatten layer
            self.flatten,
            
            # Fully connected layers
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

            # Output layer
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
            # Convolutional layers
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_rate),

            # Flatten layer
            self.flatten,

            # Fully connected layers
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

            # Output layer
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
            # Convolutional layers
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Residual blocks
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),  # This block will reduce spatial dimensions
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),  # This block will reduce spatial dimensions
            ResidualBlock(128, 128),

            # Flatten layer
            nn.AdaptiveAvgPool2d((1, 1)),  # This will adapt to any input size
            nn.Flatten(),

            # Fully connected layers
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),

            # Output layer
            nn.Linear(64, 10),
        )


class ResidualBlock(nn.Module):
    """
    A residual block for ResNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        stride (int, optional): Stride for the convolution. Defaults to 1.
        padding (int, optional): Padding for the convolution. Defaults to 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the input and output dimensions differ, adjust the input dimensions
        if in_channels != out_channels or stride != 1:
            self.adjust = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.adjust = nn.Identity()

    def forward(self, x):
        identity = self.adjust(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out
