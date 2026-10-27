# Neural Networks
In this part neural networks are introduced and studied in detail.

Classic machine-learning models (part 5) work on hand-crafted features: a domain expert decides what the
model gets to see. Neural networks instead *learn* the features as a composition of simple, differentiable
building blocks, trained end-to-end with gradient descent. This makes them the method of choice for
unstructured data – images, audio, text, graphs – and the foundation of the large language models in part 7.

The part is organised in four chapters, each with runnable scripts that finish within a minute or two on a
laptop CPU (a GPU or Apple-silicon GPU is used automatically if available):

- **Multilayer perceptrons (c00_mlp)**: the fully connected network, backpropagation written by hand in
  NumPy, the same model in PyTorch, and the practical toolbox: initialisation, activation functions,
  optimisers, regularisation, early stopping.
- **Convolutional neural networks (c01_cnn)**: weight sharing and locality for images, convolution, pooling,
  batch normalisation, data augmentation, checkpoints, and how to look inside a trained network.
- **Generative adversarial networks (c02_gans)**: learning to *generate* data with a two-player game,
  illustrated on a 2D toy distribution (where mode collapse can be seen directly) and with a DCGAN on digits.
- **Graph neural networks (c03_gnns)**: learning on graphs with message passing (GCN, GAT, GraphSAGE, GIN)
  using PyTorch Geometric.

## Tools and conventions in this part
All examples use [PyTorch](https://pytorch.org) – the de-facto standard framework in research and
increasingly in industry – and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io) for graphs.
The conda environment for the whole part is `conda_env_gzh_nn0.yml`.

A PyTorch program almost always has the same anatomy:

1. **Data**: a `Dataset` returns single examples; a `DataLoader` shuffles and stacks them into mini-batches.
2. **Model**: a subclass of `nn.Module` that declares its layers in `__init__` and the computation in `forward`.
3. **Loss and optimiser**: e.g. `nn.CrossEntropyLoss` and `torch.optim.AdamW(model.parameters())`.
4. **Training loop**: for each batch: forward pass, loss, `optimiser.zero_grad()`, `loss.backward()`
   (autograd computes all gradients), `optimiser.step()`.
5. **Evaluation**: `model.eval()` and `torch.no_grad()` (switches off dropout / batch-norm updates and
   gradient tracking).

Shared helpers live in `p06_neural_networks/utils/torch_helpers.py`: device selection (CUDA, then Apple MPS,
then CPU; override with the environment variable `TORCH_DEVICE=cpu`), seeding for reproducibility, a generic
`train_one_epoch`/`evaluate` pair and an `EarlyStopping` class. All scripts are run as modules from the
repository root, e.g. `python -m p06_neural_networks.c00_mlp.main_2_mlp_pytorch`, and write their plots,
checkpoints and downloaded data into an `outputs/` folder next to the script.
