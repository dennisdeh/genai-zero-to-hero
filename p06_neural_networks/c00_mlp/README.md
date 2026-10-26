## Environment
- Python conda environment to use: `../conda_env_gzh_nn0.yml` (part-level environment `gzh_nn0`: PyTorch, scikit-learn, matplotlib)
- Docker compose setup: None
- Environment variables: optional `TORCH_DEVICE` to force a device (e.g. `TORCH_DEVICE=cpu`); by default CUDA, then Apple MPS, then CPU is used
- Data: the two-moons dataset is generated with scikit-learn – no downloads needed

## How to run
Run from the repository root (each script finishes in well under a minute on a laptop CPU):
```bash
conda env create -f p06_neural_networks/conda_env_gzh_nn0.yml   # once
conda activate gzh_nn0
python -m p06_neural_networks.c00_mlp.main_1_mlp_numpy                    # MLP + backprop from scratch in NumPy
python -m p06_neural_networks.c00_mlp.main_2_mlp_pytorch                  # the same in PyTorch with early stopping
python -m p06_neural_networks.c00_mlp.main_2_mlp_pytorch --activation tanh --dropout 0 --weight-decay 0
python -m p06_neural_networks.c00_mlp.main_3_optimisers_regularisation    # optimiser and regularisation experiments
```
Plots and model weights are written to `p06_neural_networks/c00_mlp/outputs/`.
