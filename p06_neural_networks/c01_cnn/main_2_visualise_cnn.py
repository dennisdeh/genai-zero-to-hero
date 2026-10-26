"""
Looking inside a trained CNN: learned filters and feature maps.

What this script demonstrates:
- loading the checkpoint written by main_1_cnn_classifier.py (it trains one first if none exists)
- plotting the 3x3 kernels of the first convolutional layer (the learned "edge/stroke detectors")
- capturing intermediate activations (feature maps) with forward hooks (`register_forward_hook`)
- a simple saliency map: the gradient of the class score w.r.t. the input pixels

All figures are saved to outputs/.

How to run (from the repository root):
    python -m p06_neural_networks.c01_cnn.main_2_visualise_cnn
    python -m p06_neural_networks.c01_cnn.main_2_visualise_cnn --dataset fashion_mnist   # after training it
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from p06_neural_networks.c01_cnn.main_1_cnn_classifier import Config, load_checkpoint, make_loaders, run_training
from p06_neural_networks.utils.torch_helpers import get_device, get_output_dir


def plot_first_layer_filters(model: nn.Module, path: str) -> None:
    """Plot every kernel of the first Conv2d layer (averaged over input channels)."""
    first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    w = first_conv.weight.detach().cpu().mean(dim=1)  # (out_channels, k, k)
    n = w.shape[0]
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < n:
            lim = w[i].abs().max()
            ax.imshow(w[i], cmap="bwr", vmin=-lim, vmax=lim)  # red = positive, blue = negative weights
    fig.suptitle(f"First-layer filters ({n} kernels of size {tuple(w.shape[1:])})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def capture_feature_maps(model: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run the model on x and record the output of every ReLU via forward hooks."""
    activations: dict[str, torch.Tensor] = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            # the hook is called as hook(module, input, output) right after module.forward
            hooks.append(module.register_forward_hook(
                lambda m, inp, out, name=name: activations.__setitem__(name, out.detach().cpu())))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()  # always clean up hooks
    return activations


def plot_feature_maps(image: torch.Tensor, activations: dict[str, torch.Tensor], path: str, n_maps: int = 8) -> None:
    """Show the input image and the first n_maps channels of each recorded activation."""
    layers = list(activations.items())
    fig, axes = plt.subplots(len(layers) + 1, n_maps, figsize=(n_maps * 1.3, (len(layers) + 1) * 1.4))
    for ax in axes.ravel():
        ax.axis("off")
    axes[0, 0].imshow(image.permute(1, 2, 0).squeeze().numpy(), cmap="gray")
    axes[0, 0].set_title("input", fontsize=8)
    for row, (name, act) in enumerate(layers, start=1):
        for c in range(min(n_maps, act.shape[1])):
            axes[row, c].imshow(act[0, c].numpy(), cmap="viridis")
        axes[row, 0].set_title(f"{name} {tuple(act.shape[2:])}", fontsize=7, loc="left")
    fig.suptitle("Feature maps after each ReLU (first channels)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def saliency_map(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Gradient of the predicted class score w.r.t. the input – which pixels matter most?"""
    x = x.clone().requires_grad_(True)
    scores = model(x)
    pred = int(scores.argmax(dim=1))
    scores[0, pred].backward()
    return x.grad.abs().amax(dim=1)[0].cpu(), pred  # max over channels -> (H, W)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["digits", "fashion_mnist", "cifar10"], default="digits")
    args = parser.parse_args()
    device = get_device()
    output_dir = get_output_dir(__file__)

    # 1. Load the trained model (train it first if no checkpoint exists)
    ckpt_path = os.path.join(output_dir, f"cnn_{args.dataset}.pt")
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint at {ckpt_path} – training one first")
        run_training(Config(dataset=args.dataset))
    model, ckpt = load_checkpoint(ckpt_path, device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} ({ckpt['config']['dataset']})")

    # 2. First-layer filters
    path = os.path.join(output_dir, f"filters_{args.dataset}.png")
    plot_first_layer_filters(model, path)
    print(f"Saved {path}")

    # 3. Feature maps of one test image
    cfg = Config(**{**ckpt["config"], "channels": tuple(ckpt["config"]["channels"])})
    _, _, test_loader, meta = make_loaders(cfg)
    x, y = next(iter(test_loader))
    x0 = x[:1].to(device)
    activations = capture_feature_maps(model, x0)
    mean, std = torch.tensor(meta["mean"])[:, None, None], torch.tensor(meta["std"])[:, None, None]
    image = (x[0] * std + mean).clamp(0, 1)
    path = os.path.join(output_dir, f"feature_maps_{args.dataset}.png")
    plot_feature_maps(image, activations, path)
    print(f"Saved {path}")
    for name, act in activations.items():
        print(f"  {name:28s} shape {tuple(act.shape)}  mean activation {act.mean():.3f}  "
              f"fraction zero {float((act == 0).float().mean()):.2f}")

    # 4. Saliency maps for a few test images
    fig, axes = plt.subplots(2, 6, figsize=(11, 5.2))
    for i in range(6):
        sal, pred = saliency_map(model, x[i : i + 1].to(device))
        img = (x[i] * std + mean).clamp(0, 1).permute(1, 2, 0).squeeze().numpy()
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"true {meta['classes'][y[i]]} / pred {meta['classes'][pred]}", fontsize=7)
        axes[1, i].imshow(sal.numpy(), cmap="hot")
        axes[1, i].set_title("saliency", fontsize=7)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    fig.suptitle("Inputs (top) and saliency maps |d score / d pixel| (bottom)")
    fig.tight_layout()
    path = os.path.join(output_dir, f"saliency_{args.dataset}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")
