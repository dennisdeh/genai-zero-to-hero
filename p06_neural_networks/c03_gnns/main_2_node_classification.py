"""
Node classification with graph neural networks: GCN vs. GAT vs. GraphSAGE (and an MLP baseline).

Datasets:
- "karate" (default): Zachary's Karate Club, bundled with PyTorch Geometric (no download), 34 nodes,
  4 classes, only 4 labelled nodes (one per class) – a tiny semi-supervised problem
- "cora": the Cora citation network (2,708 papers, 1,433 bag-of-words features, 7 classes) with the
  standard Planetoid split (140 training labels); downloaded into outputs/data (requires internet)

What this script demonstrates:
- the built-in PyG layers GCNConv, GATConv (multi-head attention over neighbours) and SAGEConv
  (mean aggregation of neighbours, concatenated with the node's own features)
- full-batch semi-supervised training with masks (loss only on the labelled nodes)
- model selection on a validation mask (Cora) and results averaged over several random seeds
- an MLP that ignores the edges, to show how much the graph structure contributes
- visualisations (Karate Club): the graph coloured by true and predicted communities and the learned
  2D node embeddings (outputs/)

How to run (from the repository root):
    python -m p06_neural_networks.c03_gnns.main_2_node_classification
    python -m p06_neural_networks.c03_gnns.main_2_node_classification --dataset cora --epochs 200
"""

import argparse
import copy
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG still uses torch.jit.script internally, which recent PyTorch versions flag as deprecated
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.jit")
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub, Planetoid
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import to_networkx

from p06_neural_networks.utils.torch_helpers import count_parameters, get_device, get_output_dir, set_seed


class GNN(nn.Module):
    """
    Two-layer GNN: conv -> ELU -> dropout -> conv. `kind` selects the layer type.
    For "mlp" the layers are plain Linear layers, i.e. the graph is ignored.
    """

    def __init__(self, kind: str, in_dim: int, hidden: int, n_classes: int, dropout: float = 0.5, heads: int = 4):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        if kind == "gcn":
            self.conv1, self.conv2 = GCNConv(in_dim, hidden), GCNConv(hidden, n_classes)
        elif kind == "gat":
            # hidden layer: `heads` attention heads concatenated; output layer: a single head
            self.conv1 = GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout)
            self.conv2 = GATConv(hidden, n_classes, heads=1, dropout=dropout)
        elif kind == "sage":
            self.conv1, self.conv2 = SAGEConv(in_dim, hidden, aggr="mean"), SAGEConv(hidden, n_classes, aggr="mean")
        elif kind == "mlp":
            self.conv1, self.conv2 = nn.Linear(in_dim, hidden), nn.Linear(hidden, n_classes)
        else:
            raise ValueError(kind)

    def _layer(self, layer: nn.Module, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return layer(x) if self.kind == "mlp" else layer(x, edge_index)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Hidden node representations after the first layer."""
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.elu(self._layer(self.conv1, x, edge_index))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(self.embed(x, edge_index), p=self.dropout, training=self.training)
        return self._layer(self.conv2, h, edge_index)


def load_data(name: str) -> tuple[Data, int]:
    """Return the graph (with train/val/test masks) and the number of classes."""
    if name == "karate":
        ds = KarateClub()
        data = ds[0]
        # only 4 labelled nodes: the remaining 30 are the test set. There are too few labels for a
        # validation set, so no model selection is done (the weights after the last epoch are used).
        data.test_mask = ~data.train_mask
        return data, ds.num_classes
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "data")
    ds = Planetoid(root, "Cora")  # downloads on first use
    return ds[0], ds.num_classes


@torch.no_grad()
def accuracy(model: nn.Module, data: Data, mask: torch.Tensor) -> float:
    """Accuracy of the model on the nodes selected by mask."""
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    return (pred[mask] == data.y[mask]).float().mean().item()


def train_model(kind: str, data: Data, n_classes: int, epochs: int, seed: int, hidden: int = 16,
                lr: float = 0.01) -> tuple[nn.Module, float]:
    """
    Full-batch training. If the graph has a validation mask, the weights with the best validation
    accuracy are kept, otherwise those of the last epoch. Returns (model, test accuracy).
    """
    set_seed(seed)
    model = GNN(kind, data.num_node_features, hidden, n_classes).to(data.x.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val, best_state = -1.0, None
    has_val = "val_mask" in data
    for _ in range(epochs):
        model.train()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if has_val:
            val = accuracy(model, data, data.val_mask)
            if val > best_val:
                best_val, best_state = val, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, accuracy(model, data, data.test_mask)


@torch.no_grad()
def plot_karate(model: nn.Module, data: Data, path: str) -> None:
    """Draw the Karate Club graph with true and predicted communities, plus the hidden embeddings (PCA)."""
    model.eval()
    pred = model(data.x, data.edge_index).argmax(1).cpu().numpy()
    emb = model.embed(data.x, data.edge_index).cpu().numpy()
    emb = emb - emb.mean(0)
    emb2d = emb @ np.linalg.svd(emb, full_matrices=False)[2][:2].T  # PCA via SVD
    g = to_networkx(data.cpu(), to_undirected=True)
    pos = nx.spring_layout(g, seed=7)
    y = data.y.cpu().numpy()
    train = data.train_mask.cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, colours, title in [(axes[0], y, "true communities"), (axes[1], pred, "GCN prediction")]:
        nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.3)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colours, cmap="Set1", vmin=0, vmax=8, node_size=180,
                               edgecolors=["black" if t else "white" for t in train],
                               linewidths=[2.5 if t else 0.5 for t in train])
        ax.set_title(f"{title} (black ring = labelled node)")
        ax.axis("off")
    axes[2].scatter(emb2d[:, 0], emb2d[:, 1], c=y, cmap="Set1", vmin=0, vmax=8, s=60, edgecolors="k")
    axes[2].set_title("hidden node embeddings (PCA), coloured by true class")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["karate", "cora"], default="karate")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=5, help="number of random seeds to average over")
    args = parser.parse_args()
    device = get_device()
    output_dir = get_output_dir(__file__)

    # 1. Load the graph
    data, n_classes = load_data(args.dataset)
    data = data.to(device)
    print(f"device: {device} | {args.dataset}: {data}")
    n_val = int(data.val_mask.sum()) if "val_mask" in data else 0
    print(f"labelled training nodes {int(data.train_mask.sum())}, validation {n_val}, test {int(data.test_mask.sum())}")

    # 2. Train every model type for several seeds and compare the test accuracy
    results = {}
    for kind in ["mlp", "gcn", "gat", "sage"]:
        accs = []
        for seed in range(args.seeds):
            model, acc = train_model(kind, data, n_classes, args.epochs, seed)
            accs.append(acc)
        results[kind] = accs
        print(f"{kind.upper():5s} ({count_parameters(model):,} params): test accuracy "
              f"{np.mean(accs):.3f} ± {np.std(accs):.3f} over {args.seeds} seeds")

    # 3. Visualise the GCN on the Karate Club graph
    if args.dataset == "karate":
        model, _ = train_model("gcn", data, n_classes, args.epochs, seed=0)
        path = os.path.join(output_dir, "karate_gcn.png")
        plot_karate(model, data, path)
        print(f"Saved {path}")
