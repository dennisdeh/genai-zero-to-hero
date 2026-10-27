"""
Graph-level classification: predict a property of a whole graph with a GIN and global pooling.

Datasets:
- "synthetic" (default, offline): 600 random graphs from three generative models that a human cannot tell
  apart at a glance – Erdős–Rényi (random edges), Barabási–Albert (preferential attachment, hubs) and
  Watts–Strogatz (ring lattice with rewiring, "small world") – with similar sizes and densities.
  By default every node has the same constant feature, so the network has to learn purely from the structure
  (`--features degree` gives it one-hot encoded node degrees instead).
- "mutag": 188 molecules (mutagenic or not) from the TU collection, downloaded into outputs/data.

What this script demonstrates:
- building PyG `Data` objects from networkx graphs and a list-based dataset
- mini-batching many graphs with `torch_geometric.loader.DataLoader` (graphs are stacked into one big,
  disconnected graph; `batch` maps every node to its graph)
- the Graph Isomorphism Network layer (GINConv, Xu et al. 2019) with a learnable MLP update
- readout / global pooling (sum, mean, max) to go from node embeddings to one vector per graph
- a train/validation/test split (60/20/20): training is monitored on the validation set, the test set is
  evaluated once at the end; the pooling functions are compared over a few seeds (different splits and
  initialisations), because one small test set and one seed are not enough to rank them

How to run (from the repository root):
    python -m p06_neural_networks.c03_gnns.main_3_graph_classification
    python -m p06_neural_networks.c03_gnns.main_3_graph_classification --dataset mutag
"""

import argparse
import os
import warnings

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG still uses torch.jit.script internally, which recent PyTorch versions flag as deprecated
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.jit")
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import from_networkx

from p06_neural_networks.utils.torch_helpers import get_device, set_seed

MAX_DEGREE = 20
CLASS_NAMES = ["Erdős–Rényi", "Barabási–Albert", "Watts–Strogatz"]
POOLING = {"sum": global_add_pool, "mean": global_mean_pool, "max": global_max_pool}


def make_synthetic_dataset(n_per_class: int = 200, seed: int = 0, features: str = "constant") -> list[Data]:
    """
    Random graphs from three generators with 20-40 nodes (inclusive) and an average degree of ~4.
    features="constant": every node gets the same feature 1 – all information must come from the structure;
    features="degree": one-hot encoded node degree.
    """
    rng = np.random.default_rng(seed)
    graphs = []
    for label in range(3):
        for _ in range(n_per_class):
            n = int(rng.integers(20, 41))  # the upper bound of integers() is exclusive
            s = int(rng.integers(1_000_000))
            if label == 0:
                g = nx.gnp_random_graph(n, 4 / (n - 1), seed=s)
            elif label == 1:
                g = nx.barabasi_albert_graph(n, 2, seed=s)
            else:
                g = nx.watts_strogatz_graph(n, 4, 0.2, seed=s)
            data = from_networkx(g)
            if features == "degree":
                deg = torch.tensor([min(d, MAX_DEGREE) for _, d in g.degree()])
                data.x = F.one_hot(deg, MAX_DEGREE + 1).float()
            else:
                data.x = torch.ones(n, 1)
            data.y = torch.tensor([label])
            graphs.append(data)
    return graphs


def load_dataset(name: str, seed: int, features: str = "constant") -> tuple[list, int, int]:
    """Return (list of graphs, number of node features, number of classes)."""
    if name == "synthetic":
        graphs = make_synthetic_dataset(seed=seed, features=features)
        return graphs, graphs[0].num_node_features, 3
    from torch_geometric.datasets import TUDataset

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "data")
    ds = TUDataset(root, name="MUTAG")
    return list(ds), ds.num_node_features, ds.num_classes


class GIN(nn.Module):
    """
    GIN: h_v' = MLP((1 + eps) h_v + sum_{u in N(v)} h_u), stacked n_layers times.
    The graph representation concatenates the pooled node embeddings of every layer (jumping knowledge).
    """

    def __init__(self, in_dim: int, hidden: int, n_classes: int, n_layers: int = 3, pooling: str = "sum"):
        super().__init__()
        self.convs = nn.ModuleList()
        dim = in_dim
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp, train_eps=True))
            dim = hidden
        self.pool = POOLING[pooling]
        self.head = nn.Sequential(nn.Linear(hidden * n_layers, hidden), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        pooled = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            pooled.append(self.pool(x, batch))  # (n_graphs_in_batch, hidden)
        return self.head(torch.cat(pooled, dim=1))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Graph classification accuracy."""
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch).argmax(1)
        correct += int((pred == batch.y).sum())
        total += batch.num_graphs
    return correct / total


def run(graphs: list, in_dim: int, n_classes: int, pooling: str, epochs: int, seed: int, device: torch.device,
        verbose: bool = False) -> tuple[float, float]:
    """
    Train a GIN with the given pooling on a 60/20/20 train/validation/test split (the split depends on `seed`).
    Training is monitored on the validation set only; the test set is evaluated once, after training.
    Returns (validation accuracy, test accuracy).
    """
    set_seed(seed)
    perm = np.random.default_rng(seed).permutation(len(graphs))
    n_train, n_val = int(0.6 * len(graphs)), int(0.2 * len(graphs))
    train = [graphs[i] for i in perm[:n_train]]
    val = [graphs[i] for i in perm[n_train:n_train + n_val]]
    test = [graphs[i] for i in perm[n_train + n_val:]]
    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    val_loader, test_loader = DataLoader(val, batch_size=128), DataLoader(test, batch_size=128)
    train_eval_loader = DataLoader(train, batch_size=128)
    if verbose:
        print(f"split: {len(train)} train, {len(val)} validation, {len(test)} test graphs")
        b = next(iter(DataLoader(train, batch_size=32)))
        print(f"one mini-batch: {b.num_graphs} graphs stacked into {b.num_nodes} nodes; "
              f"batch vector (first 40 entries): {b.batch[:40].tolist()}")
    model = GIN(in_dim, 64, n_classes, pooling=pooling).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            loss = F.cross_entropy(model(batch.x, batch.edge_index, batch.batch), batch.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            train_acc, val_acc = evaluate(model, train_eval_loader, device), evaluate(model, val_loader, device)
            print(f"  epoch {epoch:3d} | last batch loss {loss.item():.4f} | train acc {train_acc:.3f} | "
                  f"val acc {val_acc:.3f}")
    return evaluate(model, val_loader, device), evaluate(model, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["synthetic", "mutag"], default="synthetic")
    parser.add_argument("--features", choices=["constant", "degree"], default="constant",
                        help="node features of the synthetic graphs")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5, help="seeds (splits/initialisations) per pooling function")
    args = parser.parse_args()
    device = get_device()

    # 1. Build / load the dataset
    graphs, in_dim, n_classes = load_dataset(args.dataset, args.seed, args.features)
    sizes = [g.num_nodes for g in graphs]
    print(f"device: {device} | {len(graphs)} graphs, {np.mean(sizes):.1f} nodes on average "
          f"({min(sizes)}-{max(sizes)}), {in_dim} node features, {n_classes} classes")
    print(f"example graph: {graphs[0]}")
    if args.dataset == "synthetic":
        print("classes: " + ", ".join(f"{i} = {name}" for i, name in enumerate(CLASS_NAMES)))

    # 2. Train a GIN with sum pooling (the most expressive readout) and show the training progress
    print("\nGIN with sum pooling")
    seeds = [args.seed + k for k in range(args.n_seeds)]
    results = {"sum": [run(graphs, in_dim, n_classes, "sum", args.epochs, seeds[0], device, verbose=True)]}
    print(f"validation accuracy {results['sum'][0][0]:.3f} | test accuracy (evaluated once) {results['sum'][0][1]:.3f}")

    # 3. Compare readout functions over several seeds (each seed: a different split and initialisation)
    for pooling in ["sum", "mean", "max"]:
        results.setdefault(pooling, [])
        for seed in seeds[len(results[pooling]):]:
            results[pooling].append(run(graphs, in_dim, n_classes, pooling, args.epochs, seed, device))
    n_test = len(graphs) - int(0.6 * len(graphs)) - int(0.2 * len(graphs))
    print(f"\nComparison of global pooling functions (mean ± std over {len(seeds)} seeds; "
          f"{n_test} test graphs per seed):")
    for pooling, res in results.items():
        val_accs, test_accs = np.array(res).T
        print(f"  {pooling:4s} pooling: val accuracy {val_accs.mean():.3f} ± {val_accs.std():.3f} | "
              f"test accuracy {test_accs.mean():.3f} ± {test_accs.std():.3f}")
    print("Differences of one or two test graphs (~1%) are within the noise: with so few graphs and seeds this\n"
          "comparison is anecdotal, not a reliable ranking of the pooling functions.")
