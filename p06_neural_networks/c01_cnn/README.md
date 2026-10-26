## Environment
- Python conda environment to use: `../conda_env_gzh_nn0.yml` (part-level environment `gzh_nn0`: PyTorch, torchvision, scikit-learn, matplotlib)
- Docker compose setup: None
- Environment variables: optional `TORCH_DEVICE` to force a device (e.g. `TORCH_DEVICE=cpu`); by default CUDA, then Apple MPS, then CPU is used
- Data: scikit-learn's bundled 8x8 digits by default (offline). `--dataset fashion_mnist` or `--dataset cifar10`
  downloads the dataset with torchvision into `c01_cnn/outputs/data` (internet access required)

## How to run
Run from the repository root:
```bash
conda activate gzh_nn0
python -m p06_neural_networks.c01_cnn.main_1_cnn_classifier       # train, evaluate and checkpoint a CNN on digits (~30 s on CPU)
python -m p06_neural_networks.c01_cnn.main_2_visualise_cnn        # filters, feature maps, saliency maps (uses the checkpoint)

# larger, downloaded datasets (use --subset for a quick CPU run, a GPU for the full data)
python -m p06_neural_networks.c01_cnn.main_1_cnn_classifier --dataset fashion_mnist --subset 10000 --epochs 5
python -m p06_neural_networks.c01_cnn.main_2_visualise_cnn --dataset fashion_mnist
python -m p06_neural_networks.c01_cnn.main_1_cnn_classifier --dataset cifar10 --epochs 30
```
Options: `--epochs`, `--batch-size`, `--lr`, `--no-augment`, `--seed`. Checkpoints (`cnn_<dataset>.pt`), the confusion
matrix, misclassified examples and the visualisations are written to `p06_neural_networks/c01_cnn/outputs/`.
