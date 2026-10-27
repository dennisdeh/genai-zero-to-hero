"""
Message passing by hand: a graph convolution (GCN) layer implemented three ways.

Data: Zachary's Karate Club graph (34 members of a club, 78 friendships, bundled with PyTorch Geometric –
no download). Node features are one-hot identity vectors; labels are 4 communities.

What this script demonstrates:
1. how PyTorch Geometric stores a graph: `x` (node features), `edge_index` (2 x E list of directed edges), `y`
2. the GCN propagation rule in dense matrix form:  H' = D^-1/2 (A + I) D^-1/2 H W
3. the same rule as message passing with `torch_geometric.nn.MessagePassing`:
       message  m_ji = norm_ji * W h_j      (computed per edge j -> i)
       aggregate  sum_j m_ji                 (scatter-add over incoming edges)
       update    h_i' = aggregate + b
4. a check that dense, hand-written and PyG's built-in `GCNConv` give identical outputs
5. training a 2-layer GCN built from the hand-written layer with only 4 labelled nodes (semi-supervised)

How to run (from the repository root):
    python -m p06_neural_networks.c03_gnns.main_1_message_passing
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG still uses torch.jit.script internally, which recent PyTorch versions flag as deprecated
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.jit")
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

from p06_neural_networks.utils.torch_helpers import get_device, set_seed


class HandmadeGCNConv(MessagePassing):
    """
    A from-scratch GCN layer (Kipf & Welling, 2017) on top of PyG's MessagePassing base class.

    `propagate()` calls message() for every edge, aggregates the messages per target node with `aggr`
    and then calls update(). Here: aggr="add" (sum), messages are normalised, linearly transformed features.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 1. add self loops so that each node also keeps its own features: A_hat = A + I
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # 2. transform the node features: W h_j
        x = self.lin(x)
        # 3. symmetric normalisation 1 / sqrt(deg(i) deg(j)) for each edge j -> i
        src, dst = edge_index
        deg = degree(dst, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        # 4. message passing: message() -> aggregate (sum) -> update()
        return self.propagate(edge_index, x=x, norm=norm) + self.bias

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """x_j are the features of the *source* node of every edge (PyG indexes them for us)."""
        return norm.view(-1, 1) * x_j


def dense_gcn(x: torch.Tensor, edge_index: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """The GCN rule with dense matrices: D^-1/2 (A + I) D^-1/2 X W^T + b (only feasible for small graphs)."""
    n = x.size(0)
    a_hat = to_dense_adj(edge_index, max_num_nodes=n)[0] + torch.eye(n)
    d_inv_sqrt = torch.diag(a_hat.sum(dim=1).pow(-0.5))
    return d_inv_sqrt @ a_hat @ d_inv_sqrt @ x @ weight.T + bias


class HandmadeGCN(nn.Module):
    """Two hand-written GCN layers: 34 -> 16 -> 4 classes."""

    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.conv1 = HandmadeGCNConv(in_dim, hidden)
        self.conv2 = HandmadeGCNConv(hidden, n_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.conv2(h, edge_index)


if __name__ == "__main__":
    set_seed(0)
    device = get_device()

    # 1. Load the graph and inspect its representation
    dataset = KarateClub()
    data = dataset[0]
    print(data)
    print(f"nodes {data.num_nodes}, directed edges {data.num_edges} (= 2 x 78 undirected), "
          f"features per node {data.num_node_features}, classes {dataset.num_classes}")
    print(f"is undirected: {data.is_undirected()}, labelled training nodes: {int(data.train_mask.sum())}")
    print(f"edge_index (first 8 edges, row 0 = source, row 1 = target):\n{data.edge_index[:, :8]}")
    neighbours_of_0 = data.edge_index[0, data.edge_index[1] == 0].tolist()
    print(f"node 0 receives messages from its {len(neighbours_of_0)} neighbours: {neighbours_of_0}")

    # 2. Compare dense matrix form, hand-written MessagePassing layer and PyG's GCNConv with shared weights
    mine = HandmadeGCNConv(data.num_node_features, 8)
    builtin = GCNConv(data.num_node_features, 8)
    with torch.no_grad():
        builtin.lin.weight.copy_(mine.lin.weight)
        mine.bias.copy_(torch.randn(8))
        builtin.bias.copy_(mine.bias)
        out_mine = mine(data.x, data.edge_index)
        out_builtin = builtin(data.x, data.edge_index)
        out_dense = dense_gcn(data.x, data.edge_index, mine.lin.weight, mine.bias)
    print("\nOne GCN layer computed three ways (output shape "
          f"{tuple(out_mine.shape)}):")
    print(f"  hand-written vs dense formula : max |diff| = {(out_mine - out_dense).abs().max():.2e}")
    print(f"  hand-written vs PyG GCNConv   : max |diff| = {(out_mine - out_builtin).abs().max():.2e}")
    assert torch.allclose(out_mine, out_dense, atol=1e-5) and torch.allclose(out_mine, out_builtin, atol=1e-5)

    # 3. Semi-supervised node classification with the hand-written layers
    data = data.to(device)
    model = HandmadeGCN(data.num_node_features, 16, dataset.num_classes).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    test_mask = ~data.train_mask  # evaluate on every node that was not labelled
    for epoch in range(201):
        model.train()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])  # loss on 4 nodes only
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
            acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
            print(f"epoch {epoch:3d} | train loss {loss.item():.4f} | accuracy on the 30 unlabelled nodes {acc:.3f}")
