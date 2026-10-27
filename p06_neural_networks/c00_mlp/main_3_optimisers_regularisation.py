"""
Experiments with optimisers and regularisation for an MLP (PyTorch).

What this script demonstrates:
1. Optimisers: the same network trained with SGD, SGD + momentum, RMSprop and Adam – compare the
   training-loss curves (speed of convergence, noise).
2. Regularisation: a deliberately over-sized network on a small, noisy dataset overfits. Compare
   no regularisation, dropout, weight decay (L2) and both – look at the validation loss and at how
   smooth the decision boundaries are.

Plots are saved to outputs/.

How to run (from the repository root):
    python -m p06_neural_networks.c00_mlp.main_3_optimisers_regularisation
"""

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

from p06_neural_networks.c00_mlp.main_2_mlp_pytorch import MLP
from p06_neural_networks.utils.torch_helpers import evaluate, get_device, get_output_dir, set_seed, train_one_epoch


def moons_tensors(n_samples: int, noise: float, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """Two-moons data split 50/50 into train and validation, standardised, as tensors."""
    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=0.5, random_state=seed, stratify=y)
    mean, std = x_tr.mean(axis=0), x_tr.std(axis=0)
    f = lambda a: torch.tensor((a - mean) / std, dtype=torch.float32)  # noqa: E731
    return f(x_tr), torch.tensor(y_tr), f(x_va), torch.tensor(y_va)


def compare_optimisers(device: torch.device, epochs: int = 60) -> dict[str, list[float]]:
    """Train identical MLPs (same initial weights) with different optimisers; return loss curves."""
    x_tr, y_tr, x_va, y_va = moons_tensors(1000, 0.2)
    optimiser_factories = {
        "SGD (lr=0.05)": lambda p: torch.optim.SGD(p, lr=0.05),
        "SGD + momentum 0.9": lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
        "RMSprop (lr=1e-3)": lambda p: torch.optim.RMSprop(p, lr=1e-3),
        "Adam (lr=1e-3)": lambda p: torch.optim.Adam(p, lr=1e-3),
    }
    curves = {}
    for name, make_opt in optimiser_factories.items():
        set_seed(0)  # same initial weights and same batch order for every optimiser
        model = MLP(2, 64, 2, n_hidden_layers=2, activation="relu", dropout=0.0).to(device)
        loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=32, shuffle=True)
        opt = make_opt(model.parameters())
        losses = [train_one_epoch(model, loader, nn.CrossEntropyLoss(), opt, device)[0] for _ in range(epochs)]
        _, val_acc = evaluate(model, DataLoader(TensorDataset(x_va, y_va), batch_size=512),
                              nn.CrossEntropyLoss(), device)
        curves[name] = losses
        print(f"  {name:22s} final train loss {losses[-1]:.4f} | val acc {val_acc:.3f}")
    return curves


def compare_regularisation(device: torch.device, epochs: int = 500) -> tuple[dict, tuple]:
    """
    Train a large MLP on only 80 noisy points with different regularisers (full-batch AdamW) and
    measure the loss on a large validation set (2,000 points from the same distribution).
    Returns the trained models with their validation-loss curves, and the training data for plotting.
    """
    x_tr, y_tr = make_moons(n_samples=80, noise=0.25, random_state=3)
    x_va, y_va = make_moons(n_samples=2000, noise=0.25, random_state=100)
    mean, std = x_tr.mean(axis=0), x_tr.std(axis=0)
    f = lambda a: torch.tensor((a - mean) / std, dtype=torch.float32, device=device)  # noqa: E731
    x_tr, x_va = f(x_tr), f(x_va)
    y_tr, y_va = torch.tensor(y_tr, device=device), torch.tensor(y_va, device=device)
    # note: AdamW's decoupled weight decay shrinks the weights by lr * weight_decay per step,
    # so with lr = 1e-3 fairly large values are needed for a visible effect
    settings = {
        "no regularisation": dict(dropout=0.0, weight_decay=0.0),
        "dropout 0.5": dict(dropout=0.5, weight_decay=0.0),
        "weight decay 5": dict(dropout=0.0, weight_decay=5.0),
        "dropout + weight decay": dict(dropout=0.5, weight_decay=5.0),
    }
    results = {}
    loss_fn = nn.CrossEntropyLoss()
    for name, cfg in settings.items():
        set_seed(0)
        model = MLP(2, 256, 2, n_hidden_layers=3, activation="relu", dropout=cfg["dropout"]).to(device)
        # AdamW applies decoupled weight decay: w <- w - lr * wd * w at every step
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=cfg["weight_decay"])
        val_losses = []
        for _ in range(epochs):
            model.train()
            loss = loss_fn(model(x_tr), y_tr)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                val_losses.append(loss_fn(model(x_va), y_va).item())
        with torch.no_grad():
            train_acc = (model(x_tr).argmax(1) == y_tr).float().mean().item()
            val_acc = (model(x_va).argmax(1) == y_va).float().mean().item()
        print(f"  {name:24s} train acc {train_acc:.3f} | val acc {val_acc:.3f} | "
              f"final val loss {val_losses[-1]:.3f} (min {min(val_losses):.3f})")
        results[name] = (model, val_losses)
    return results, (x_tr.cpu(), y_tr.cpu())


@torch.no_grad()
def plot_regularisation(results: dict, data: tuple, device: torch.device, path: str) -> None:
    """Decision boundaries (top row) and validation-loss curves (bottom) for each regulariser."""
    x, y = data
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device)
    fig = plt.figure(figsize=(16, 8))
    for i, (name, (model, _)) in enumerate(results.items()):
        ax = fig.add_subplot(2, len(results), i + 1)
        model.eval()
        probs = torch.softmax(model(grid), 1)[:, 1].cpu().numpy().reshape(xx.shape)
        ax.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.6)
        ax.contour(xx, yy, probs, levels=[0.5], colors="k")
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap="RdBu_r", edgecolors="k", s=15)
        ax.set_title(name)
    ax = fig.add_subplot(2, 1, 2)
    for name, (_, val_losses) in results.items():
        ax.plot(val_losses, label=name)
    ax.set_xlabel("epoch (full batch)")
    ax.set_ylabel("validation loss")
    ax.set_ylim(0, 1.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    # note: get_device() also enables torch.set_flush_denormal(True) on CPU – without it, the tiny gradients
    # of the over-fitted network below can become "denormal" floats, which some CPUs process much more slowly.
    # If training is slow on a shared machine, first suspect thread oversubscription (set OMP_NUM_THREADS=1).
    device = get_device()
    output_dir = get_output_dir(__file__)
    print(f"device: {device}")

    # 1. Compare optimisers
    print("\n1. Optimisers (same initial weights, 60 epochs, batch size 32)")
    curves = compare_optimisers(device)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, losses in curves.items():
        ax.plot(losses, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (log scale)")
    ax.set_title("Optimiser comparison on two moons")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "optimisers.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")

    # 2. Compare regularisation techniques on a small dataset with an over-sized network
    print("\n2. Regularisation (MLP 3x256 on 80 training points, 2,000 validation points)")
    results, data = compare_regularisation(device)
    path = os.path.join(output_dir, "regularisation.png")
    plot_regularisation(results, data, device, path)
    print(f"Saved {path}")
