"""
The same MLP as in main_1_mlp_numpy.py, now written with PyTorch.

What this script demonstrates:
- defining a model as an `nn.Module` (configurable depth, width, activation and dropout)
- `TensorDataset` + `DataLoader` for shuffling and mini-batching
- the standard training loop (forward -> loss -> zero_grad -> backward -> step), autograd does the backprop
- train / validation / test split, AdamW optimiser with weight decay, a learning-rate scheduler
- early stopping on the validation loss (restoring the best weights)
- saving the trained weights (state_dict) and plotting the decision boundary + learning curves to outputs/

How to run (from the repository root):
    python -m p06_neural_networks.c00_mlp.main_2_mlp_pytorch
    python -m p06_neural_networks.c00_mlp.main_2_mlp_pytorch --activation tanh --dropout 0.0 --weight-decay 0
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from p06_neural_networks.utils.torch_helpers import (
    EarlyStopping,
    count_parameters,
    evaluate,
    get_device,
    get_output_dir,
    set_seed,
    train_one_epoch,
)

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU, "sigmoid": nn.Sigmoid}


class MLP(nn.Module):
    """
    Fully connected network: [Linear -> activation -> Dropout] x n_hidden_layers -> Linear.

    The output layer returns raw scores (logits); the softmax is folded into `nn.CrossEntropyLoss`
    for numerical stability.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_hidden_layers: int = 2,
                 activation: str = "relu", dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(dim, hidden_dim), ACTIVATIONS[activation](), nn.Dropout(dropout)]
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.reset_parameters(activation)

    def reset_parameters(self, activation: str) -> None:
        """He (Kaiming) initialisation for ReLU-like activations, Xavier (Glorot) otherwise."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation in ("relu", "leaky_relu", "gelu"):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_loaders(batch_size: int, seed: int) -> tuple[DataLoader, DataLoader, DataLoader, tuple]:
    """Create train/validation/test DataLoaders for the two-moons dataset (standardised)."""
    x, y = make_moons(n_samples=1500, noise=0.25, random_state=seed)
    x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, test_size=0.4, random_state=seed, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)

    def to_ds(a: np.ndarray, b: np.ndarray) -> TensorDataset:
        return TensorDataset(torch.tensor((a - mean) / std, dtype=torch.float32), torch.tensor(b, dtype=torch.long))

    g = torch.Generator().manual_seed(seed)  # reproducible shuffling
    train_loader = DataLoader(to_ds(x_train, y_train), batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(to_ds(x_val, y_val), batch_size=512)
    test_loader = DataLoader(to_ds(x_test, y_test), batch_size=512)
    return train_loader, val_loader, test_loader, (mean, std)


@torch.no_grad()
def plot_decision_boundary(model: nn.Module, loader: DataLoader, history: dict, device: torch.device,
                           best_epoch: int, path: str) -> None:
    """Plot predicted class probabilities on a grid (left) and the learning curves (right)."""
    model.eval()
    x = torch.cat([b[0] for b in loader]).numpy()
    y = torch.cat([b[1] for b in loader]).numpy()
    xx, yy = np.meshgrid(np.linspace(x[:, 0].min() - 0.5, x[:, 0].max() + 0.5, 300),
                         np.linspace(x[:, 1].min() - 0.5, x[:, 1].max() + 0.5, 300))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device)
    probs = torch.softmax(model(grid), dim=1)[:, 1].cpu().numpy().reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.6)
    axes[0].contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=1.5)
    axes[0].scatter(x[:, 0], x[:, 1], c=y, cmap="RdBu_r", edgecolors="k", s=15)
    axes[0].set_title("PyTorch MLP: decision boundary (test set)")
    axes[1].plot(history["train_loss"], label="train loss")
    axes[1].plot(history["val_loss"], label="validation loss")
    axes[1].axvline(best_epoch, color="grey", linestyle=":", label=f"best epoch ({best_epoch})")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cross-entropy")
    axes[1].legend()
    axes[1].set_title("Learning curves with early stopping")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Command line options to experiment with architecture and regularisation."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--activation", choices=list(ACTIVATIONS), default="relu")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    output_dir = get_output_dir(__file__)
    print(f"PyTorch {torch.__version__} | device: {device}")

    # 1. Load data into DataLoaders
    train_loader, val_loader, test_loader, _ = make_loaders(args.batch_size, args.seed)
    print(f"Batches per epoch: {len(train_loader)} (batch size {args.batch_size})")

    # 2. Define the model, loss function, optimiser and learning-rate scheduler
    model = MLP(2, args.hidden_dim, 2, args.layers, args.activation, args.dropout).to(device)
    print(model)
    print(f"Trainable parameters: {count_parameters(model):,}")
    loss_fn = nn.CrossEntropyLoss()
    # AdamW = Adam with decoupled weight decay (L2-type regularisation)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=10)
    stopper = EarlyStopping(patience=args.patience)

    # 3. Training loop with validation and early stopping
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimiser, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if epoch % 20 == 0:
            print(f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.3f} | lr {optimiser.param_groups[0]['lr']:.1e}")
        if stopper.step(val_loss, model, epoch):
            print(f"Early stopping at epoch {epoch}: no improvement for {args.patience} epochs")
            break
    stopper.restore_best(model)
    print(f"Restored weights of best epoch {stopper.best_epoch} (val loss {stopper.best_loss:.4f})")

    # 4. Final evaluation on the held-out test set
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"Test loss {test_loss:.4f} | test accuracy {test_acc:.3f}")

    # 5. Save the weights and visualise the result
    weights_path = os.path.join(output_dir, "mlp_moons.pt")
    torch.save(model.state_dict(), weights_path)
    plot_path = os.path.join(output_dir, "mlp_pytorch_decision_boundary.png")
    plot_decision_boundary(model, test_loader, history, device, stopper.best_epoch, plot_path)
    print(f"Saved weights to {weights_path}\nSaved plot to {plot_path}")
