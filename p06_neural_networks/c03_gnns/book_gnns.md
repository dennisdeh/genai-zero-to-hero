## Graph Neural Networks

Many data sets are naturally graphs: social networks, citation networks, molecules (atoms and bonds), road
networks, knowledge graphs, interactions between proteins, users and products in a recommender system. A graph
$\mathcal{G} = (\mathcal{V}, \mathcal{E})$ has nodes with feature vectors $\mathbf{x}_v$ and edges between them; it has no
fixed size, no natural ordering of the nodes and no grid structure, so MLPs and CNNs cannot be applied
directly. Graph neural networks (GNNs) generalise the idea of the convolution – shared weights applied to a
local neighbourhood – to arbitrary graphs.

Typical tasks are **node classification** (the topic of a paper in a citation network), **link prediction**
(will two users become friends?) and **graph classification/regression** (is a molecule toxic?).

| Script | What it shows |
|---|---|
| `main_1_message_passing.py` | how PyG stores graphs; a GCN layer written by hand with `MessagePassing`, checked against the matrix formula and `GCNConv` |
| `main_2_node_classification.py` | GCN vs. GAT vs. GraphSAGE vs. an MLP on Karate Club (offline) or Cora (download) |
| `main_3_graph_classification.py` | graph classification with GIN and global pooling on synthetic random graphs (offline) or MUTAG (download) |

The examples use [PyTorch Geometric](https://pytorch-geometric.readthedocs.io) (PyG), the most widely used GNN
library for PyTorch.

### Graphs in PyTorch Geometric
A graph is a `torch_geometric.data.Data` object with

- `x`: node feature matrix of shape $[N, F]$,
- `edge_index`: a $[2, E]$ integer tensor listing the edges as (source, target) pairs – a sparse *COO*
  representation; an undirected edge is stored in both directions,
- `y`: labels (per node or per graph), plus optional masks such as `train_mask`.

`main_1_message_passing.py` prints these for **Zachary's Karate Club** (1977): 34 members of a university
karate club, 78 friendships, and – in PyG's version – four communities as labels. Only four nodes, one per
community, are labelled for training; the task is to infer the community of the other 30 from the friendships.
This *semi-supervised* setting, where the graph structure propagates information from few labelled nodes, is
where GNNs shine.

### Message passing
Almost all GNNs fit into the **message passing** framework (Gilmer et al. 2017). In every layer each node
collects messages from its neighbours $\mathcal{N}(i)$, aggregates them with a permutation-invariant function
(sum, mean, max) and updates its own representation:
$$ \mathbf{h}_i^{(l+1)} = \gamma\Big(\mathbf{h}_i^{(l)},\; \bigoplus_{j \in \mathcal{N}(i)} \phi\big(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, \mathbf{e}_{ji}\big)\Big), $$
with a *message* function $\phi$, an *aggregation* $\bigoplus$ and an *update* function $\gamma$ (learnable networks).
Because the aggregation does not depend on the order of the neighbours, the output is independent of how the
nodes are numbered (permutation equivariance). After $L$ layers a node's representation depends on its
$L$-hop neighbourhood – the graph analogue of a receptive field. Most GNNs use only 2–3 layers: with many
layers all node representations converge to similar vectors (*over-smoothing*).

PyG's `MessagePassing` base class implements this pattern: a layer calls `self.propagate(edge_index, ...)`,
which calls the layer's `message()` for every edge – arguments with the suffix `_j` (e.g. `x_j`) are
automatically gathered from the source node of each edge and `_i` from the target – then aggregates the messages
per target node with `aggr="add"`, `"mean"` or `"max"` (an efficient scatter operation), and finally calls `update()`.

### Graph convolutional networks (GCN)
Kipf and Welling (2017) derived a simple and effective layer as a first-order approximation of spectral graph
convolutions. With adjacency matrix $A$, self-loops added $\hat{A} = A + I$ (so a node keeps its own features), and
the degree matrix $\hat{D}_{ii} = \sum_j \hat{A}_{ij}$, the **GCN propagation rule** is
$$ H^{(l+1)} = \sigma\Big(\hat{D}^{-1/2}\, \hat{A}\, \hat{D}^{-1/2}\, H^{(l)}\, W^{(l)}\Big). $$
Per node this reads
$$ \mathbf{h}_i^{(l+1)} = \sigma\Big( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\hat{d}_i \hat{d}_j}}\, W^{(l)} \mathbf{h}_j^{(l)} \Big): $$
a symmetrically normalised weighted sum of the transformed features of the neighbours and the node itself.
It is not an average: the weights $1/\sqrt{\hat{d}_i \hat{d}_j}$ of a node do not sum to one (on Karate Club
the row sums of $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ range from about 0.63 to 2.0) – they are close to one only
when the neighbours have about the same degree as the node. The normalisation therefore does not make the
scale exactly independent of the degree, but it keeps it under control: without it, high-degree nodes would get
sums that grow linearly with their degree, whereas the normalised matrix has largest eigenvalue 1, so stacking
layers does not blow up the features. It also down-weights messages from hubs. `HandmadeGCNConv` in `main_1_message_passing.py`
implements exactly this: add self-loops, apply the linear map $W$, compute `norm` $= 1/\sqrt{\hat{d}_i\hat{d}_j}$ per edge,
and let `message()` return `norm * x_j`, which `propagate()` sums per node. The script verifies that the dense
matrix formula, the hand-written layer and PyG's `GCNConv` with the same weights give identical outputs (up to
floating-point rounding), and then trains a two-layer GCN built from the hand-written layer on Karate Club.

The dense formula needs an $N \times N$ matrix and is only feasible for small graphs; message passing works on
the edge list, so the cost scales with the number of edges $E$ – this is what makes GNNs applicable to graphs
with millions of nodes (combined with neighbour sampling, as in GraphSAGE).

### Graph attention networks (GAT)
In a GCN the weight of a neighbour is fixed by the degrees. Veličković et al. (2018) let the network *learn* how
much attention to pay to each neighbour:
$$ \alpha_{ij} = \operatorname{softmax}_{j \in \mathcal{N}(i)\cup\{i\}}\Big(\operatorname{LeakyReLU}\big(\mathbf{a}^\top [W\mathbf{h}_i \,\Vert\, W\mathbf{h}_j]\big)\Big), \qquad \mathbf{h}_i' = \sigma\Big(\sum_{j} \alpha_{ij} W \mathbf{h}_j\Big), $$
where $\Vert$ denotes concatenation and $\mathbf{a}$ is a learnable vector. As in transformers, several attention
*heads* run in parallel and their outputs are concatenated (hidden layers) or averaged (output layer). The
attention mechanism is a direct relative of the self-attention in transformers – a transformer can be seen as a
GAT on a fully connected graph.

### GraphSAGE
Hamilton, Ying and Leskovec (2017) designed GraphSAGE for large, evolving graphs (inductive learning: embedding
nodes that were not seen during training). The layer aggregates the neighbours (mean, max-pool or LSTM) and
combines the result with the node's own representation through a separate weight matrix:
$$ \mathbf{h}_i' = \sigma\Big(W_1 \mathbf{h}_i + W_2 \cdot \operatorname{mean}_{j \in \mathcal{N}(i)} \mathbf{h}_j\Big). $$
Training on mini-batches of sampled neighbourhoods (`torch_geometric.loader.NeighborLoader`) lets it scale to
graphs with billions of edges, e.g. in recommender systems.

### Node classification results
`main_2_node_classification.py` trains each model five times with different seeds (full-batch, Adam, weight
decay, dropout) and reports the mean and standard deviation of the test accuracy. On Karate Club typical results
are:

- **MLP**: about 0.3 – barely better than guessing. The node features are one-hot identity vectors, so a model
  that ignores the edges has nothing to generalise from.
- **GCN**: about 0.77, **GAT**: about 0.85 – the graph structure carries the community information.
- **GraphSAGE**: about 0.5. Its separate self-weight $W_1$ can memorise the four labelled identity vectors, and
  with only one labelled node per class the neighbourhood term is not learned well enough – a reminder that
  the best architecture depends on the data.

The script also draws the graph coloured by true and predicted communities, and the hidden node embeddings
projected to 2D, where the communities form clusters even though only four nodes were labelled.

On **Cora** (`--dataset cora`; 2,708 papers, 1,433 bag-of-words features, 7 topics, 140 training labels) the
classic picture appears: the MLP reaches just over 50 % test accuracy, while GCN, GAT and GraphSAGE all reach about
80 %. Here the best epoch is selected on the 500 validation nodes.

### Graph classification
To classify whole graphs, the node embeddings must be combined into one vector per graph with a permutation-
invariant **readout** (global pooling): sum, mean or max over the nodes, followed by an MLP.
`main_3_graph_classification.py` uses the **Graph Isomorphism Network** (GIN, Xu et al. 2019):
$$ \mathbf{h}_i' = \operatorname{MLP}\Big((1+\epsilon)\, \mathbf{h}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j\Big). $$
Xu et al. showed that message-passing GNNs are at most as powerful as the Weisfeiler–Lehman graph isomorphism
test at distinguishing graphs, and that GIN – with *sum* aggregation and an MLP update – reaches this upper
bound. Mean and max aggregators lose information: mean cannot count (a node with two red neighbours looks
like one with one red neighbour) and max ignores multiplicities altogether.

The offline dataset consists of 600 random graphs from three generative models with similar sizes (20–40 nodes)
and average degree (about 4): Erdős–Rényi (independent random edges), Barabási–Albert (preferential attachment,
producing hubs) and Watts–Strogatz (a ring lattice with some rewired edges, "small world"). By default every
node has the same constant feature, so the network must discover the structural differences itself; with
`--features degree` it gets the one-hot degree as input. Mini-batching works differently from images: PyG's
`DataLoader` stacks the graphs of a batch into one large disconnected graph and creates a `batch` vector that
maps every node to its graph, which the pooling functions use. The script uses *jumping knowledge* (the pooled
embeddings of all layers are concatenated) and a 60/20/20 train/validation/test split: training progress is
printed on the validation set, and the test set is evaluated only once, after training. With sum pooling the GIN
reaches about 98 % validation accuracy after 10 epochs and 98.3 % test accuracy at the end. The script then
compares sum, mean and max pooling over five seeds (each with a different split and initialisation): all three
reach 97–99 % mean test accuracy (sum 98.3 ± 0.9 %, mean 97.7 ± 1.3 %, max 98.8 ± 1.2 %). With 120 test graphs
per seed one graph is 0.8 %, so these differences are within the noise – on this easy task the comparison is
anecdotal and does not show the advantage of sum pooling expected from theory (a single seed can suggest any
order). With `--dataset mutag` the script classifies molecules from the TU collection.

### Beyond this chapter
Edge features (bond types) enter through the message function; heterogeneous graphs have several node and edge
types (`HeteroData` in PyG); link prediction scores pairs of node embeddings; graph transformers combine
message passing with global attention. GNNs are used in drug discovery, materials science, traffic forecasting
(Google Maps ETAs), weather prediction and recommender systems.

**Things to try:** use `aggr="mean"` or `"max"` in `HandmadeGCNConv` (without the normalisation) and compare;
add a third GCN layer and watch the accuracy on Cora; plot the GAT attention weights
(call a `GATConv` layer as `conv(x, edge_index, return_attention_weights=True)`); run the graph classification with `--features degree`.
