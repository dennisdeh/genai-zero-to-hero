"""
A small DCGAN (deep convolutional GAN) that generates handwritten digits.

Data: scikit-learn's bundled 8x8 digits (1,797 images, no download), upsampled to 16x16 and scaled to [-1, 1].

What this script demonstrates:
- the DCGAN architecture guidelines (Radford et al. 2015): transposed convolutions in G, strided convolutions
  in D, batch norm (not in D's input layer / G's output layer), ReLU in G, LeakyReLU in D, tanh output,
  Adam(lr=2e-4, beta1=0.5), weights initialised from N(0, 0.02)
- the non-saturating GAN training loop with one-sided label smoothing
- image grids generated from a fixed noise vector at several epochs (outputs/dcgan_epoch_*.png)
- a cheap sample-quality / mode-collapse check: a classifier trained on real digits labels the generated
  images; we report the class histogram, its entropy and the mean confidence (an "Inception-Score-lite")
- latent-space interpolation between two noise vectors

How to run (from the repository root):
    python -m p06_neural_networks.c02_gans.main_2_dcgan_digits
    python -m p06_neural_networks.c02_gans.main_2_dcgan_digits --epochs 150
"""

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from p06_neural_networks.utils.torch_helpers import count_parameters, get_device, get_output_dir, set_seed

IMG_SIZE = 16


def load_digit_images() -> tuple[torch.Tensor, np.ndarray]:
    """Return the digits as a tensor (N, 1, 16, 16) in [-1, 1] and the labels."""
    digits = load_digits()
    x = torch.tensor(digits.images / 16.0, dtype=torch.float32)[:, None]  # (N, 1, 8, 8) in [0, 1]
    x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return x * 2 - 1, digits.target  # tanh range


class Generator(nn.Module):
    """z (z_dim) -> 128x2x2 -> 64x4x4 -> 32x8x8 -> 1x16x16 with transposed convolutions (base = 32)."""

    def __init__(self, z_dim: int = 64, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            # project and reshape: a transposed conv on a 1x1 "image" with kernel 2 gives 2x2
            nn.ConvTranspose2d(z_dim, base * 4, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4
            nn.BatchNorm2d(base * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8
            nn.BatchNorm2d(base),
            nn.ReLU(True),
            nn.ConvTranspose2d(base, 1, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z[:, :, None, None])


class Discriminator(nn.Module):
    """1x16x16 -> 32x8x8 -> 64x4x4 -> 128x2x2 -> logit, with strided convolutions (base = 32)."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1),  # no batch norm on the input layer
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 4, 1, 2, 1, 0),  # 2x2 -> 1x1 logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def dcgan_init(m: nn.Module) -> None:
    """DCGAN weight initialisation: conv weights ~ N(0, 0.02), batch-norm scale ~ N(1, 0.02)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def save_grid(images: torch.Tensor, path: str, nrow: int = 8, title: str = "") -> None:
    """Save a grid of images in [-1, 1]."""
    images = ((images.detach().cpu() + 1) / 2).clamp(0, 1)
    n = images.shape[0]
    rows = int(np.ceil(n / nrow))
    fig, axes = plt.subplots(rows, nrow, figsize=(nrow * 0.9, rows * 0.9 + 0.4))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i, 0], cmap="gray_r", vmin=0, vmax=1)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fit_digit_classifier(x: torch.Tensor, y: np.ndarray) -> LogisticRegression:
    """A simple classifier on real (16x16) digits, used to label generated samples."""
    clf = LogisticRegression(max_iter=2000)
    clf.fit(x.view(len(x), -1).numpy(), y)
    return clf


def sample_quality(clf: LogisticRegression, samples: torch.Tensor) -> tuple[np.ndarray, float, float]:
    """
    Class histogram of generated samples, entropy of that histogram (max log(10) = 2.30 means
    all digits are produced equally often – no mode collapse) and mean classifier confidence.
    """
    proba = clf.predict_proba(samples.detach().cpu().view(len(samples), -1).numpy())
    hist = np.bincount(proba.argmax(1), minlength=10) / len(proba)
    entropy = float(-(hist[hist > 0] * np.log(hist[hist > 0])).sum())
    return hist, entropy, float(proba.max(1).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()
    output_dir = get_output_dir(__file__)
    print(f"device: {device}")

    # 1. Load data
    x, y = load_digit_images()
    loader = DataLoader(TensorDataset(x), batch_size=args.batch_size, shuffle=True, drop_last=True)
    save_grid(x[:64], os.path.join(output_dir, "dcgan_real.png"), title="real digits (16x16)")
    clf = fit_digit_classifier(x, y)
    print(f"Real data: {tuple(x.shape)}, classifier accuracy on real data: "
          f"{clf.score(x.view(len(x), -1).numpy(), y):.3f}")

    # 2. Build the networks and optimisers
    G, D = Generator(args.z_dim).to(device), Discriminator().to(device)
    G.apply(dcgan_init)
    D.apply(dcgan_init)
    print(f"Generator: {count_parameters(G):,} parameters | Discriminator: {count_parameters(D):,} parameters")
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    z_fixed = torch.randn(64, args.z_dim, device=device)
    grid_epochs = {0, args.epochs // 4, args.epochs // 2, args.epochs - 1}

    # 3. Training loop
    t0 = time.time()
    for epoch in range(args.epochs):
        d_losses, g_losses, d_real_acc = [], [], []
        for (real,) in loader:
            real = real.to(device)
            b = real.size(0)
            # 3a. discriminator step on a real and a fake batch
            fake = G(torch.randn(b, args.z_dim, device=device))
            d_real, d_fake = D(real), D(fake.detach())
            loss_d = bce(d_real, torch.full_like(d_real, 0.9)) + bce(d_fake, torch.zeros_like(d_fake))
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            # 3b. generator step with the non-saturating loss -log D(G(z))
            d_fake = D(fake)
            loss_g = bce(d_fake, torch.ones_like(d_fake))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            d_losses.append(loss_d.item())
            g_losses.append(loss_g.item())
            d_real_acc.append((d_real > 0).float().mean().item())
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d} | loss D {np.mean(d_losses):.3f} | loss G {np.mean(g_losses):.3f} | "
                  f"D accuracy on real {np.mean(d_real_acc):.2f} | {time.time() - t0:.0f}s")
        if epoch in grid_epochs:
            G.eval()  # use batch-norm running statistics for sampling
            with torch.no_grad():
                save_grid(G(z_fixed), os.path.join(output_dir, f"dcgan_epoch_{epoch:03d}.png"),
                          title=f"generated digits, epoch {epoch}")
            G.train()

    # 4. Evaluate sample quality and diversity
    G.eval()
    with torch.no_grad():
        samples = G(torch.randn(2000, args.z_dim, device=device))
    hist, entropy, conf = sample_quality(clf, samples)
    print("\nClass histogram of 2000 generated digits:", " ".join(f"{d}:{h:.2f}" for d, h in enumerate(hist)))
    print(f"Entropy of the histogram: {entropy:.2f} (max {np.log(10):.2f}) | mean classifier confidence {conf:.2f}")

    # 5. Latent-space interpolation between two random noise vectors
    with torch.no_grad():
        z0, z1 = torch.randn(2, args.z_dim, device=device)
        alphas = torch.linspace(0, 1, 10, device=device)[:, None]
        rows = [G((1 - alphas) * z0 + alphas * z1)]
        for _ in range(3):
            z0, z1 = torch.randn(2, args.z_dim, device=device)
            rows.append(G((1 - alphas) * z0 + alphas * z1))
    path = os.path.join(output_dir, "dcgan_interpolation.png")
    save_grid(torch.cat(rows), path, nrow=10, title="linear interpolation in latent space (one row per pair)")
    torch.save(G.state_dict(), os.path.join(output_dir, "dcgan_generator.pt"))
    print(f"Saved image grids, interpolation and generator weights to {output_dir}")
