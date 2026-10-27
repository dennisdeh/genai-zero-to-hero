## Environment
- Python conda environment to use: `../conda_env_gzh_nn0.yml` (part-level environment `gzh_nn0`: PyTorch, PyTorch Geometric, networkx, matplotlib)
- Docker compose setup: None
- Environment variables: optional `TORCH_DEVICE` to force a device (e.g. `TORCH_DEVICE=cpu`); by default CUDA, then Apple MPS, then CPU is used
- Data: Karate Club is bundled with PyTorch Geometric and the synthetic graphs are generated with networkx (offline).
  `--dataset cora` (Planetoid) and `--dataset mutag` (TU collection) download into `c03_gnns/outputs/data` (internet access required)

## How to run
Run from the repository root:
```bash
conda activate gzh_nn0
python -m p06_neural_networks.c03_gnns.main_1_message_passing          # GCN layer by hand with MessagePassing (seconds)
python -m p06_neural_networks.c03_gnns.main_2_node_classification      # GCN vs. GAT vs. GraphSAGE vs. MLP on Karate Club
python -m p06_neural_networks.c03_gnns.main_2_node_classification --dataset cora   # Cora citation network (download, a few minutes on CPU)
python -m p06_neural_networks.c03_gnns.main_3_graph_classification     # GIN graph classification on synthetic graphs
python -m p06_neural_networks.c03_gnns.main_3_graph_classification --features degree
python -m p06_neural_networks.c03_gnns.main_3_graph_classification --dataset mutag # molecules (download)
```
The Karate Club visualisation and downloaded datasets are written to `p06_neural_networks/c03_gnns/outputs/`.
