"""
A generative adversarial network (GAN) on a 2D toy distribution: a ring of 8 Gaussians.

Because the data is two-dimensional we can *see* what the generator learns, and directly measure
mode collapse (the generator covering only some of the 8 modes).

What this script demonstrates:
- generator G: noise z ~ N(0, I) -> point in R^2, discriminator D: point -> real/fake logit (both MLPs)
- the alternating training loop: k discriminator steps, then one generator step
- the original minimax generator loss  log(1 - D(G(z)))  vs. the non-saturating loss  -log D(G(z))
- common training tricks: Adam with beta1 = 0.5, LeakyReLU in D, one-sided label smoothing, moderate learning rates
- mode collapse: without the tricks (large learning rate, default Adam momentum, no label smoothing) the
  generator collapses onto a single mode and then "hops" between modes
- quality metrics for the toy problem: number of modes covered and fraction of "high-quality" samples
- snapshots of the generated distribution during training, saved to outputs/

How to run (from the repository root):
    python -m p06_neural_networks.c02_gans.main_1_gan_2d                     # compare three set-ups
    python -m p06_neural_networks.c02_gans.main_1_gan_2d --configs non_saturating --steps 5000
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from p06_neural_networks.utils.torch_helpers import get_device, get_output_dir, set_seed

N_MODES, RADIUS, MODE_STD = 8, 2.0, 0.05

# training set-ups compared by default (all share the architecture and the random seed)
CONFIGS = {
    "non_saturating": dict(kind="non_saturating"),  # with the tricks
    "minimax": dict(kind="minimax"),  # with the tricks
    # no tricks: 5x larger learning rates, default Adam momentum (beta1 = 0.9), no label smoothing
    "no_tricks": dict(kind="non_saturating", lr_g=5e-3, lr_d=5e-3, beta1=0.9, label_smoothing=0.0),
}
CENTRES = torch.tensor([[RADIUS * np.cos(2 * np.pi * k / N_MODES), RADIUS * np.sin(2 * np.pi * k / N_MODES)]
                        for k in range(N_MODES)], dtype=torch.float32)


def sample_real(n: int, device: torch.device) -> torch.Tensor:
    """Sample n points from the ring of 8 Gaussians."""
    idx = torch.randint(0, N_MODES, (n,))
    return (CENTRES[idx] + MODE_STD * torch.randn(n, 2)).to(device)


def mlp(in_dim: int, out_dim: int, hidden: int, act: type[nn.Module]) -> nn.Sequential:
    """Three-hidden-layer MLP used for both networks."""
    return nn.Sequential(nn.Linear(in_dim, hidden), act(), nn.Linear(hidden, hidden), act(),
                         nn.Linear(hidden, hidden), act(), nn.Linear(hidden, out_dim))


def mode_metrics(samples: torch.Tensor, min_share: float = 0.2) -> tuple[int, float]:
    """
    A sample is "high quality" if it lies within 3 standard deviations of some mode centre.
    A mode is "covered" if the high-quality samples near it make up at least `min_share` of the 1/8 of *all*
    samples a perfect generator would put there (default: 2.5% of all samples). Counting against all samples,
    not only the high-quality ones, means a generator that produces mostly garbage cannot cover a mode with a
    handful of lucky samples. Report both numbers: coverage measures diversity, the high-quality fraction
    measures fidelity.
    Returns (number of covered modes, fraction of high-quality samples).
    """
    d = torch.cdist(samples.cpu(), CENTRES)  # (n, 8) distances to every centre
    dmin, nearest = d.min(dim=1)
    good = dmin < 3 * MODE_STD
    counts = torch.bincount(nearest[good], minlength=N_MODES)  # high-quality samples per mode
    covered = int((counts >= min_share * len(samples) / N_MODES).sum())
    return covered, float(good.float().mean())


def generator_loss(d_fake_logits: torch.Tensor, kind: str) -> torch.Tensor:
    """
    Generator loss. With D(x) = sigmoid(logit):
    - "minimax":         mean log(1 - D(G(z)))   (minimised; gradients vanish when D confidently rejects fakes)
    - "non_saturating":  mean -log D(G(z))       (Goodfellow et al. 2014 recommendation; strong early gradients)
    """
    if kind == "minimax":
        return -F.binary_cross_entropy_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))
    return F.binary_cross_entropy_with_logits(d_fake_logits, torch.ones_like(d_fake_logits))


def train_gan(kind: str, steps: int, device: torch.device, seed: int = 0, z_dim: int = 2, d_steps: int = 1,
              g_steps: int = 1, lr_d: float = 1e-3, lr_g: float = 1e-3, beta1: float = 0.5,
              label_smoothing: float = 0.1, n_snapshots: int = 6, name: str = "") -> tuple[list, list]:
    """Train a GAN with the given generator loss. Returns snapshots [(step, samples)] and metric history."""
    set_seed(seed)
    name = name or kind
    G = mlp(z_dim, 2, 128, nn.ReLU).to(device)
    D = mlp(2, 1, 128, lambda: nn.LeakyReLU(0.2)).to(device)
    # Adam with beta1 = 0.5 (less momentum) is a classic GAN trick: the "target" moves as the other player learns
    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, 0.999))
    batch = 256
    z_fixed = torch.randn(2000, z_dim, device=device)  # fixed noise to visualise progress
    snapshot_steps = set(np.linspace(0, steps - 1, n_snapshots).astype(int).tolist())
    snapshots, history = [], []

    for step in range(steps):
        # 1. Discriminator: maximise log D(x) + log(1 - D(G(z)))  <=>  minimise BCE with labels real=1, fake=0
        for _ in range(d_steps):
            real = sample_real(batch, device)
            fake = G(torch.randn(batch, z_dim, device=device)).detach()  # no gradient into G here
            d_real, d_fake = D(real), D(fake)
            real_targets = torch.full_like(d_real, 1.0 - label_smoothing)  # one-sided label smoothing
            loss_d = (F.binary_cross_entropy_with_logits(d_real, real_targets)
                      + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

        # 2. Generator: fool the discriminator
        for _ in range(g_steps):
            fake = G(torch.randn(batch, z_dim, device=device))
            loss_g = generator_loss(D(fake), kind)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        # 3. Monitoring
        if step % 250 == 0 or step in snapshot_steps:
            with torch.no_grad():
                samples = G(z_fixed)
            covered, quality = mode_metrics(samples)
            history.append((step, covered, quality))
            if step in snapshot_steps:
                snapshots.append((step, samples.cpu().numpy()))
            if step % 500 == 0 or step == steps - 1:
                print(f"  [{name}] step {step:5d} | loss D {loss_d.item():.3f} | loss G {loss_g.item():.3f} | "
                      f"modes covered {covered}/8 | high-quality samples {quality:.2f}")
    return snapshots, history


def plot_snapshots(all_snapshots: dict[str, list], path: str) -> None:
    """Grid of generated samples over training: one row per generator loss."""
    rows, cols = len(all_snapshots), len(next(iter(all_snapshots.values())))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6), squeeze=False)
    real = sample_real(2000, torch.device("cpu")).numpy()
    for r, (kind, snaps) in enumerate(all_snapshots.items()):
        for c, (step, s) in enumerate(snaps):
            ax = axes[r, c]
            ax.scatter(real[:, 0], real[:, 1], s=2, c="lightgrey")
            ax.scatter(s[:, 0], s[:, 1], s=2, c="tab:red", alpha=0.5)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"step {step}", fontsize=9)
        axes[r, 0].set_ylabel(kind)
    fig.suptitle("Generated samples (red) vs. real data (grey)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--d-steps", type=int, default=1, help="discriminator updates per generator update")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = get_device()
    output_dir = get_output_dir(__file__)
    print(f"device: {device}")

    # 1. Train one GAN per set-up (same seed and architecture)
    snapshots, histories = {}, {}
    for name in args.configs:
        print(f"\nTraining GAN set-up '{name}': {CONFIGS[name]}")
        snapshots[name], histories[name] = train_gan(steps=args.steps, device=device, seed=args.seed,
                                                     d_steps=args.d_steps, name=name, **CONFIGS[name])

    # 2. Plot the evolution of the generated distribution
    path = os.path.join(output_dir, "gan_2d_snapshots.png")
    plot_snapshots(snapshots, path)
    print(f"\nSaved {path}")

    # 3. Plot mode coverage over training
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, h in histories.items():
        steps, covered, quality = zip(*h)
        axes[0].plot(steps, covered, label=name)
        axes[1].plot(steps, quality, label=name)
    axes[0].set_ylabel("modes covered (of 8)")
    axes[1].set_ylabel("fraction of high-quality samples")
    for ax in axes:
        ax.set_xlabel("generator step")
        ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "gan_2d_mode_coverage.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Saved {path}")
