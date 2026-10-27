"""
A multilayer perceptron (MLP) written from scratch in NumPy to understand backpropagation.

What this script demonstrates:
- the forward pass of a fully connected network (affine maps + non-linear activations)
- the backward pass: gradients of the loss w.r.t. every weight, derived by hand with the chain rule
- a numerical gradient check (finite differences) to verify the hand-written backward pass
- mini-batch gradient descent on the toy "two moons" dataset (scikit-learn, no download)
- He/Xavier initialisation and its effect on training
- a plot of the learned decision boundary and the loss curve (saved to outputs/)

Network: 2 inputs -> 32 hidden (ReLU) -> 32 hidden (ReLU) -> 1 output (sigmoid), binary cross-entropy loss.

How to run (from the repository root):
    python -m p06_neural_networks.c00_mlp.main_1_mlp_numpy
"""

import os

import matplotlib

matplotlib.use("Agg")  # render to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


# ----------------------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------------------
def relu(z: np.ndarray) -> np.ndarray:
    """Rectified linear unit, max(0, z)."""
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU w.r.t. its input (1 where z > 0, else 0)."""
    return (z > 0).astype(z.dtype)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def bce_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Binary cross-entropy: -mean(y log p + (1-y) log(1-p))."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class NumpyMLP:
    """
    A small MLP with ReLU hidden layers and a sigmoid output for binary classification.

    Parameters are stored in lists W[l], b[l]; the forward pass caches the pre-activations z[l]
    and activations a[l] that the backward pass needs.
    """

    def __init__(self, layer_sizes: list[int], init: str = "he", seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            if init == "he":  # suited for ReLU: Var(w) = 2 / fan_in
                std = np.sqrt(2.0 / fan_in)
            elif init == "xavier":  # suited for tanh/sigmoid: Var(w) = 2 / (fan_in + fan_out)
                std = np.sqrt(2.0 / (fan_in + fan_out))
            elif init == "small":  # deliberately bad: tiny weights -> vanishing signal
                std = 0.01
            else:
                raise ValueError(f"Unknown initialisation: {init}")
            self.W.append(rng.normal(0.0, std, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. Returns the output probabilities, shape (n, 1)."""
        self.a = [x]  # a[0] is the input
        self.z = []
        n_layers = len(self.W)
        for l in range(n_layers):
            z = self.a[-1] @ self.W[l] + self.b[l]  # affine map
            self.z.append(z)
            a = sigmoid(z) if l == n_layers - 1 else relu(z)  # non-linearity
            self.a.append(a)
        return self.a[-1]

    def backward(self, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Backward pass (backpropagation) for the BCE loss, using the caches from `forward`.

        For a sigmoid output with BCE, dL/dz_out simplifies to (p - y) / n.
        Then, layer by layer from the top:
            dW[l] = a[l]^T  @ delta[l]
            db[l] = sum over the batch of delta[l]
            delta[l-1] = (delta[l] @ W[l]^T) * relu'(z[l-1])
        """
        n = y.shape[0]
        grads_W = [np.zeros_like(W) for W in self.W]
        grads_b = [np.zeros_like(b) for b in self.b]
        delta = (self.a[-1] - y) / n  # dL/dz of the output layer
        for l in reversed(range(len(self.W))):
            grads_W[l] = self.a[l].T @ delta
            grads_b[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ self.W[l].T) * relu_grad(self.z[l - 1])
        return grads_W, grads_b

    def step(self, grads_W: list[np.ndarray], grads_b: list[np.ndarray], lr: float, weight_decay: float = 0.0):
        """Plain gradient descent update, optionally with L2 weight decay on the weights."""
        for l in range(len(self.W)):
            self.W[l] -= lr * (grads_W[l] + weight_decay * self.W[l])
            self.b[l] -= lr * grads_b[l]


# ----------------------------------------------------------------------------------------------
# Gradient check, training and plotting
# ----------------------------------------------------------------------------------------------
def gradient_check(model: NumpyMLP, x: np.ndarray, y: np.ndarray, eps: float = 1e-6, n_checks: int = 10) -> float:
    """
    Compare analytical gradients from `backward` with central finite differences
    (L(w + eps) - L(w - eps)) / (2 eps) for a few random weights. Returns the max relative error.
    """
    rng = np.random.default_rng(1)
    model.forward(x)
    grads_W, _ = model.backward(y)
    max_rel_err = 0.0
    for _ in range(n_checks):
        l = rng.integers(len(model.W))
        i, j = rng.integers(model.W[l].shape[0]), rng.integers(model.W[l].shape[1])
        old = model.W[l][i, j]
        model.W[l][i, j] = old + eps
        loss_plus = bce_loss(model.forward(x), y)
        model.W[l][i, j] = old - eps
        loss_minus = bce_loss(model.forward(x), y)
        model.W[l][i, j] = old
        numerical = (loss_plus - loss_minus) / (2 * eps)
        analytical = grads_W[l][i, j]
        rel_err = abs(numerical - analytical) / max(1e-8, abs(numerical) + abs(analytical))
        max_rel_err = max(max_rel_err, rel_err)
    return max_rel_err


def train(
    model: NumpyMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 0.1,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Mini-batch gradient descent. Returns the train/validation loss history."""
    rng = np.random.default_rng(seed)
    history = {"train_loss": [], "val_loss": []}
    n = x_train.shape[0]
    for epoch in range(epochs):
        perm = rng.permutation(n)  # reshuffle each epoch
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            model.forward(x_train[idx])
            grads_W, grads_b = model.backward(y_train[idx])
            model.step(grads_W, grads_b, lr=lr, weight_decay=weight_decay)
        history["train_loss"].append(bce_loss(model.forward(x_train), y_train))
        history["val_loss"].append(bce_loss(model.forward(x_val), y_val))
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d} | train loss {history['train_loss'][-1]:.4f} | "
                  f"val loss {history['val_loss'][-1]:.4f}")
    return history


def accuracy(model: NumpyMLP, x: np.ndarray, y: np.ndarray) -> float:
    """Classification accuracy with a 0.5 threshold."""
    return float(((model.forward(x) > 0.5) == y).mean())


def plot_results(model: NumpyMLP, x: np.ndarray, y: np.ndarray, histories: dict[str, dict], path: str) -> None:
    """Plot the decision boundary of the model and the loss curves for each initialisation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # decision boundary on a grid
    xx, yy = np.meshgrid(np.linspace(x[:, 0].min() - 0.5, x[:, 0].max() + 0.5, 300),
                         np.linspace(x[:, 1].min() - 0.5, x[:, 1].max() + 0.5, 300))
    probs = model.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[0].contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.6)
    axes[0].contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=1.5)
    axes[0].scatter(x[:, 0], x[:, 1], c=y.ravel(), cmap="RdBu_r", edgecolors="k", s=15)
    axes[0].set_title("NumPy MLP: decision boundary (He init.)")
    for name, h in histories.items():
        axes[1].plot(h["train_loss"], label=f"{name} – train")
        axes[1].plot(h["val_loss"], "--", label=f"{name} – val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("binary cross-entropy")
    axes[1].set_title("Loss curves for different initialisations")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data: two interleaving half circles with noise (not linearly separable)
    x, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    y = y.reshape(-1, 1).astype(float)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=42)
    # standardise the features using training statistics only
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    x_train, x_val = (x_train - mean) / std, (x_val - mean) / std
    print(f"Train: {x_train.shape}, validation: {x_val.shape}")

    # 2. Verify the hand-written backward pass with a numerical gradient check
    layer_sizes = [2, 32, 32, 1]
    model = NumpyMLP(layer_sizes, init="he", seed=0)
    rel_err = gradient_check(model, x_train[:64], y_train[:64])
    print(f"Gradient check – max relative error: {rel_err:.2e} ({'OK' if rel_err < 1e-4 else 'CHECK BACKWARD PASS'})")

    # 3. Train with different initialisations to see why initialisation matters
    histories = {}
    models = {}
    for init in ["he", "xavier", "small"]:
        print(f"\nTraining with '{init}' initialisation")
        models[init] = NumpyMLP(layer_sizes, init=init, seed=0)
        histories[init] = train(models[init], x_train, y_train, x_val, y_val, epochs=300)
        print(f"  -> train acc {accuracy(models[init], x_train, y_train):.3f}, "
              f"val acc {accuracy(models[init], x_val, y_val):.3f}")

    # 4. Visualise the decision boundary and loss curves
    path = os.path.join(OUTPUT_DIR, "mlp_numpy_decision_boundary.png")
    plot_results(models["he"], x_val, y_val, histories, path)
    print(f"\nSaved plot to {path}")
