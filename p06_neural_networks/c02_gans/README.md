## Environment
- Python conda environment to use: `../conda_env_gzh_nn0.yml` (part-level environment `gzh_nn0`: PyTorch, scikit-learn, matplotlib)
- Docker compose setup: None
- Environment variables: optional `TORCH_DEVICE` to force a device (e.g. `TORCH_DEVICE=cpu`); by default CUDA, then Apple MPS, then CPU is used
- Data: generated (ring of Gaussians) and scikit-learn's bundled digits – no downloads needed

## How to run
Run from the repository root (each script takes about a minute on a laptop CPU):
```bash
conda activate gzh_nn0
python -m p06_neural_networks.c02_gans.main_1_gan_2d            # 2D toy GAN: non-saturating vs. minimax vs. no tricks (mode collapse)
python -m p06_neural_networks.c02_gans.main_1_gan_2d --configs minimax --d-steps 5 --steps 4000
python -m p06_neural_networks.c02_gans.main_2_dcgan_digits      # DCGAN generating handwritten digits
python -m p06_neural_networks.c02_gans.main_2_dcgan_digits --epochs 150 --z-dim 32
```
Sample snapshots, mode-coverage curves, image grids per epoch, the latent interpolation and the generator weights
are written to `p06_neural_networks/c02_gans/outputs/`.
