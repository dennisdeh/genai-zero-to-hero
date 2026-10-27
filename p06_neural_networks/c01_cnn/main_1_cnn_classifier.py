"""
Convolutional neural network (CNN) image classifier in PyTorch.

What this script demonstrates:
- convolution, batch normalisation, ReLU and max-pooling blocks, global average pooling and a linear head
- data augmentation (random rotations/shifts, horizontal flips for natural images) applied on the fly
- a training loop with validation, a cosine learning-rate schedule and early stopping
- evaluation with a classification report, a confusion matrix and a gallery of misclassified images
- saving a full checkpoint (weights, optimiser state, epoch, config) and loading it back into a fresh model

Datasets (offline-first):
- "digits" (default): scikit-learn's bundled 8x8 handwritten digits (1,797 images), no download needed
- "fashion_mnist" / "cifar10": downloaded with torchvision into outputs/data (requires internet);
  use --subset to train on a fraction for a quick CPU run

How to run (from the repository root):
    python -m p06_neural_networks.c01_cnn.main_1_cnn_classifier
    python -m p06_neural_networks.c01_cnn.main_1_cnn_classifier --dataset fashion_mnist --subset 10000 --epochs 5
"""

import argparse
import os
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from p06_neural_networks.utils.torch_helpers import (
    EarlyStopping,
    count_parameters,
    evaluate,
    get_device,
    get_output_dir,
    predict,
    set_seed,
    train_one_epoch,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


@dataclass
class Config:
    """Hyper-parameters of a run (stored inside the checkpoint)."""

    dataset: str = "digits"
    epochs: int = 40
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 1e-4
    channels: tuple = (32, 64, 128)
    dropout: float = 0.3
    augment: bool = True
    subset: Optional[int] = None
    patience: int = 10
    seed: int = 42


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
class ImageTensorDataset(Dataset):
    """Images (N, C, H, W) and labels held in memory, with an optional per-sample transform."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform: Optional[Callable] = None):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.images[idx]
        if self.transform is not None:
            x = self.transform(x)  # random augmentation: a new variant each time the image is drawn
        return x, self.labels[idx]


def load_images(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Return train and test images as float arrays (N, C, H, W) in [0, 1], labels and class names.
    """
    if cfg.dataset == "digits":
        digits = load_digits()
        x = (digits.images / 16.0).astype(np.float32)[:, None]  # (1797, 1, 8, 8)
        x_tr, x_te, y_tr, y_te = train_test_split(x, digits.target, test_size=0.2, random_state=cfg.seed,
                                                  stratify=digits.target)
        return x_tr, y_tr, x_te, y_te, [str(i) for i in range(10)]

    from torchvision import datasets  # only needed for the downloadable datasets

    root = os.path.join(OUTPUT_DIR, "data")
    ds_cls = {"fashion_mnist": datasets.FashionMNIST, "cifar10": datasets.CIFAR10}[cfg.dataset]
    train, test = ds_cls(root, train=True, download=True), ds_cls(root, train=False, download=True)

    def to_array(ds) -> tuple[np.ndarray, np.ndarray]:
        data = ds.data.numpy() if torch.is_tensor(ds.data) else np.asarray(ds.data)
        data = data[:, None] if data.ndim == 3 else data.transpose(0, 3, 1, 2)  # -> (N, C, H, W)
        return (data / 255.0).astype(np.float32), np.asarray(ds.targets)

    x_tr, y_tr = to_array(train)
    x_te, y_te = to_array(test)
    if cfg.subset:  # smaller random subset for fast CPU runs
        rng = np.random.default_rng(cfg.seed)
        idx_tr = rng.choice(len(x_tr), min(cfg.subset, len(x_tr)), replace=False)
        idx_te = rng.choice(len(x_te), min(cfg.subset // 5, len(x_te)), replace=False)
        x_tr, y_tr, x_te, y_te = x_tr[idx_tr], y_tr[idx_tr], x_te[idx_te], y_te[idx_te]
    return x_tr, y_tr, x_te, y_te, list(train.classes)


def make_loaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Split off a validation set, normalise with training statistics and build the DataLoaders."""
    x_tr, y_tr, x_te, y_te, classes = load_images(cfg)
    x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, test_size=0.2, random_state=cfg.seed, stratify=y_tr)
    mean = x_tr.mean(axis=(0, 2, 3)).tolist()  # per-channel statistics
    std = x_tr.std(axis=(0, 2, 3)).tolist()
    normalise = v2.Normalize(mean, std)

    # augmentation is applied to the training set only (and before normalisation)
    augmentations = []
    if cfg.augment:
        augmentations.append(v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)))
        if cfg.dataset == "cifar10":  # horizontal flips are fine for natural images, not for digits
            augmentations.append(v2.RandomHorizontalFlip())
    train_tf = v2.Compose(augmentations + [normalise])

    def ds(x: np.ndarray, y: np.ndarray, tf: Callable) -> ImageTensorDataset:
        return ImageTensorDataset(torch.from_numpy(x), torch.from_numpy(np.asarray(y)).long(), tf)

    g = torch.Generator().manual_seed(cfg.seed)
    loaders = (
        DataLoader(ds(x_tr, y_tr, train_tf), batch_size=cfg.batch_size, shuffle=True, generator=g),
        DataLoader(ds(x_va, y_va, normalise), batch_size=256),
        DataLoader(ds(x_te, y_te, normalise), batch_size=256),
    )
    meta = {"classes": classes, "mean": mean, "std": std, "in_channels": x_tr.shape[1],
            "image_size": tuple(x_tr.shape[2:])}
    print(f"Dataset '{cfg.dataset}': train {len(x_tr)}, val {len(x_va)}, test {len(x_te)}, "
          f"image shape {tuple(x_tr.shape[1:])}, {len(classes)} classes")
    return *loaders, meta


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Conv(3x3) -> BatchNorm -> ReLU -> Conv(3x3) -> BatchNorm -> ReLU -> optional MaxPool(2x2)."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            # bias is redundant before batch norm (BN has its own shift parameter)
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) if pool else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    """
    A small VGG-style CNN: a stack of ConvBlocks (each halves the resolution, except the last),
    global average pooling (works for any input size) and a linear classification head.
    """

    def __init__(self, in_channels: int, n_classes: int, channels: tuple = (32, 64, 128), dropout: float = 0.3):
        super().__init__()
        blocks, ch = [], in_channels
        for i, out_ch in enumerate(channels):
            blocks.append(ConvBlock(ch, out_ch, pool=i < len(channels) - 1))
            ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(ch, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ----------------------------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------------------------
def save_checkpoint(path: str, model: nn.Module, optimiser: torch.optim.Optimizer, epoch: int, cfg: Config,
                    meta: dict) -> None:
    """Save everything needed to resume training or to run inference later."""
    torch.save({"model_state": model.state_dict(), "optimiser_state": optimiser.state_dict(), "epoch": epoch,
                "config": asdict(cfg), "meta": meta}, path)


def load_checkpoint(path: str, device: torch.device) -> tuple[SimpleCNN, dict]:
    """Rebuild the model from the stored config and load its weights. Returns (model, checkpoint dict)."""
    # weights_only=True only unpickles tensors and basic python types (safe for untrusted files)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg, meta = ckpt["config"], ckpt["meta"]
    model = SimpleCNN(meta["in_channels"], len(meta["classes"]), tuple(cfg["channels"]), cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


# ----------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], path: str) -> None:
    """Save a confusion matrix (rows: true class, columns: predicted class)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=classes).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_misclassified(loader: DataLoader, y_true: np.ndarray, y_pred: np.ndarray, meta: dict, path: str,
                       n: int = 16) -> None:
    """Show up to n misclassified test images with their true and predicted labels."""
    images = torch.cat([b[0] for b in loader])
    mean, std = torch.tensor(meta["mean"])[:, None, None], torch.tensor(meta["std"])[:, None, None]
    wrong = np.where(y_true != y_pred)[0][:n]
    if len(wrong) == 0:
        return
    cols = 8
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        img = (images[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img.squeeze(), cmap="gray_r" if img.shape[2] == 1 else None)
        ax.set_title(f"true {meta['classes'][y_true[i]]}\npred {meta['classes'][y_pred[i]]}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Training pipeline
# ----------------------------------------------------------------------------------------------
def run_training(cfg: Config) -> str:
    """Train, evaluate and save a CNN according to `cfg`. Returns the checkpoint path."""
    set_seed(cfg.seed)
    device = get_device()
    output_dir = get_output_dir(__file__)
    print(f"PyTorch {torch.__version__} | device: {device}")

    # 1. Load data
    train_loader, val_loader, test_loader, meta = make_loaders(cfg)

    # 2. Define model, loss, optimiser, scheduler
    model = SimpleCNN(meta["in_channels"], len(meta["classes"]), cfg.channels, cfg.dropout).to(device)
    print(f"SimpleCNN with {count_parameters(model):,} trainable parameters")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.epochs)
    stopper = EarlyStopping(patience=cfg.patience)

    # 3. Train with validation and early stopping
    for epoch in range(cfg.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimiser, device)
        va_loss, va_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        if stopper.step(va_loss, model, epoch):
            print(f"Early stopping at epoch {epoch}")
            break
    stopper.restore_best(model)

    # 4. Save a checkpoint
    ckpt_path = os.path.join(output_dir, f"cnn_{cfg.dataset}.pt")
    save_checkpoint(ckpt_path, model, optimiser, stopper.best_epoch, cfg, meta)
    print(f"Saved checkpoint (best epoch {stopper.best_epoch}) to {ckpt_path}")

    # 5. Load the checkpoint into a fresh model and evaluate on the test set
    model, _ = load_checkpoint(ckpt_path, device)
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    y_true, y_pred = predict(model, test_loader, device)
    print(f"\nReloaded model – test loss {test_loss:.4f} | test accuracy {test_acc:.3f}")
    print(classification_report(y_true, y_pred, target_names=meta["classes"], digits=3))

    # 6. Visualise errors
    plot_confusion_matrix(y_true, y_pred, meta["classes"], os.path.join(output_dir, f"confusion_{cfg.dataset}.png"))
    plot_misclassified(test_loader, y_true, y_pred, meta, os.path.join(output_dir, f"misclassified_{cfg.dataset}.png"))
    print(f"Plots saved to {output_dir}")
    return ckpt_path


def parse_args() -> Config:
    """Read the command line into a Config."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["digits", "fashion_mnist", "cifar10"], default="digits")
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--subset", type=int, default=None, help="number of training images (downloaded datasets)")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=Config.seed)
    a = p.parse_args()
    return Config(dataset=a.dataset, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, subset=a.subset,
                  augment=not a.no_augment, seed=a.seed)


if __name__ == "__main__":
    run_training(parse_args())
