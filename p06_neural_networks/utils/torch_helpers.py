"""
Shared PyTorch helpers for the neural network chapters (p06).

Contents:
- get_device(): pick the best available accelerator (cuda -> mps -> cpu)
- set_seed(): make runs reproducible (python, numpy, torch)
- get_output_dir(): create/return an `outputs/` folder next to a script
- train_one_epoch() / evaluate(): a generic supervised training loop for classifiers
- EarlyStopping: stop training when the validation loss stops improving and remember the best weights
- count_parameters(): number of trainable parameters of a model

Usage (from the repository root):
    from p06_neural_networks.utils.torch_helpers import get_device, set_seed
"""

import copy
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_device(prefer: Optional[str] = None) -> torch.device:
    """
    Return the best available torch device.

    The order of preference is CUDA (NVIDIA GPUs), MPS (Apple silicon) and finally CPU.
    The environment variable `TORCH_DEVICE` (or the argument `prefer`) can force a device, e.g. "cpu".

    On CPU, "denormal" (subnormal) floats are flushed to zero: once a network fits its training data,
    gradients and activations can become so tiny that they are stored as denormals, and arithmetic on
    them is 10-100x slower on many x86 CPUs. Flushing them changes the results negligibly.
    """
    forced = prefer or os.environ.get("TORCH_DEVICE")
    if forced:
        device = torch.device(forced)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cpu":
        torch.set_flush_denormal(True)
    return device


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (CPU and all GPUs) for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_output_dir(script_file: str) -> str:
    """Return (and create) the `outputs/` folder next to the given script file (pass `__file__`)."""
    path = os.path.join(os.path.dirname(os.path.abspath(script_file)), "outputs")
    os.makedirs(path, exist_ok=True)
    return path


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train a classifier for one epoch.

    The canonical PyTorch loop: forward pass -> loss -> zero gradients -> backward pass -> optimiser step.
    Returns the mean loss and the accuracy over the epoch.
    """
    model.train()  # enables dropout and batch-norm statistics updates
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> tuple[float, float]:
    """Evaluate a classifier (no gradients, eval mode). Returns the mean loss and the accuracy."""
    model.eval()  # disables dropout, batch-norm uses running statistics
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (true labels, predicted labels) as numpy arrays for all batches of a loader."""
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(device))
        y_true.append(y.numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


class EarlyStopping:
    """
    Stop training when the monitored validation loss has not improved for `patience` epochs.

    The weights of the best epoch are kept in memory and can be restored with `restore_best()`.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.best_state: Optional[dict] = None
        self.counter = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Update with the latest validation loss. Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Load the weights of the best epoch into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
