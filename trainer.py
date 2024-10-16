# trainer.py

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from typing import Tuple, List, Dict


class DigitClassifier:
    """
    Class for training and evaluating a digit classifier model.

    Args:
        model (nn.Module): The neural network model to be trained.
        device (str, optional): Device to run the model on. Defaults to 'cuda'.
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
    ):
        self.model = model
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=1,
            gamma=0.95,
        )

    def train(self, dataloader: DataLoader, verbose: bool = True) -> None:
        size = len(dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                if verbose:
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, dataloader: DataLoader, verbose: bool = True) -> None:
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        if verbose:
            print(
                f"Test Error: \n Accuracy: {(100*correct):>0.1f}%,"
                f" Avg loss: {test_loss:>8f} \n"
            )

    def train_model(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        epochs: int = 3,
        verbose: bool = True,
    ) -> None:
        for t in range(epochs):
            if verbose:
                print(f"Epoch {t+1}\n-------------------------------")
            self.train(train_dataloader, verbose)
            self.test(test_dataloader, verbose)
            self.scheduler.step()

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )

    def evaluate_dataset(
        self,
        dataloader: DataLoader,
    ) -> Tuple[Dict[int, List[int]], int, int]:
        self.model.eval()
        total_correct = 0
        total_samples = 0
        digit_stats = {i: [0, 0] for i in range(10)}

        for (x, y) in dataloader:
            with torch.no_grad():
                x = x.to(self.device)
                pred = self.model(x)
                predicted = pred.argmax(1)
                for actual, pred in zip(y, predicted):
                    actual_item = actual.item()
                    pred_item = pred.item()
                    digit_stats[actual_item][1] += 1
                    if pred_item == actual_item:
                        digit_stats[actual_item][0] += 1
                        total_correct += 1

                total_samples += len(x)

        return digit_stats, total_correct, total_samples
