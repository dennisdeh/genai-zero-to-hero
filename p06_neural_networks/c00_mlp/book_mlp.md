## Multilayer Perceptrons

The multilayer perceptron (MLP), also called a fully connected or feed-forward network, is the simplest
deep-learning model and contains every ingredient of the larger architectures that follow: layers of
learnable affine maps, non-linear activation functions, a loss function, gradients computed with
backpropagation and an optimiser that updates the parameters.

This chapter contains three scripts, run from the repository root:

| Script | What it shows |
|---|---|
| `main_1_mlp_numpy.py` | an MLP from scratch in NumPy: forward pass, hand-derived backward pass, gradient check, initialisation |
| `main_2_mlp_pytorch.py` | the same model with PyTorch: `nn.Module`, `DataLoader`, training loop, validation, early stopping |
| `main_3_optimisers_regularisation.py` | SGD vs. momentum vs. RMSprop vs. Adam; dropout and weight decay against overfitting |

All three use the *two moons* dataset from scikit-learn (two interleaving half circles with noise), a
two-dimensional problem that is not linearly separable – so we need a non-linear model – and whose decision
boundary can be plotted.

### From a neuron to a network
A single artificial neuron computes a weighted sum of its inputs plus a bias and passes it through a
non-linear *activation function* $\sigma$:
$$ a = \sigma(\mathbf{w}^\top \mathbf{x} + b). $$
A *layer* applies many neurons to the same input at once, which is a matrix–vector product. Stacking $L$
layers gives the MLP; with $\mathbf{a}^{(0)} = \mathbf{x}$,
$$ \mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \qquad \mathbf{a}^{(l)} = \sigma\big(\mathbf{z}^{(l)}\big), \qquad l = 1, \dots, L. $$
Here $\mathbf{z}^{(l)}$ are the *pre-activations* and $\mathbf{a}^{(l)}$ the *activations*. For classification
the last layer produces one score (*logit*) per class, which a softmax turns into probabilities,
$p_k = e^{z_k} / \sum_j e^{z_j}$ (or a sigmoid for a single binary output).

Without the non-linearity the composition of affine maps would again be affine – the network would be no
more powerful than linear regression. Common activation functions:

- **sigmoid** $\sigma(z) = 1/(1+e^{-z})$ and **tanh**: smooth, bounded; they *saturate* for large $|z|$, where
  their derivative is close to zero, which slows learning in deep networks (vanishing gradients).
- **ReLU** $\max(0, z)$: cheap, does not saturate for $z>0$; the default choice for hidden layers. Units can
  "die" if their input is always negative; **Leaky ReLU** ($\max(\alpha z, z)$) and **GELU** (smooth, used in
  transformers) are popular variants.

**Universal approximation.** Why do we expect this to work at all? The universal approximation theorem
states that an MLP with a single hidden layer and a non-polynomial activation $\sigma$ can approximate any
continuous function on a compact set $K \subset \mathbb{R}^d$ to arbitrary accuracy:
for every $\varepsilon > 0$ there are $N$, $\mathbf{w}_i$, $b_i$, $v_i$ such that
$$ \sup_{\mathbf{x} \in K} \Big| f(\mathbf{x}) - \sum_{i=1}^{N} v_i\, \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i) \Big| < \varepsilon. $$
Cybenko (1989) proved this for sigmoidal activations and Hornik (1991) for bounded, non-constant ones;
Leshno, Lin, Pinkus and Schocken (1993) showed that for continuous (more generally, locally bounded and
piecewise continuous) activations the condition "not a polynomial" is both sufficient and necessary – which
covers unbounded activations such as ReLU.
The theorem guarantees *existence* only: it says nothing about how many neurons are needed, or whether
gradient descent finds the weights, or whether the result generalises to new data. In practice, deeper
networks represent many functions with far fewer parameters than shallow ones, which is one reason why
"deep" learning works.

### Learning: loss, gradient descent and backpropagation
Training means minimising the average loss over the training set,
$\mathcal{L}(\theta) = \frac{1}{n}\sum_i \ell(f_\theta(\mathbf{x}_i), y_i)$, with respect to all parameters
$\theta = \{W^{(l)}, \mathbf{b}^{(l)}\}$. For classification $\ell$ is the *cross-entropy*
$-\log p_{y}$ (for binary problems $-y\log p - (1-y)\log(1-p)$). Gradient descent repeatedly takes a small step
against the gradient, $\theta \leftarrow \theta - \eta\, \nabla_\theta \mathcal{L}$, with learning rate $\eta$.
*Stochastic* (mini-batch) gradient descent estimates the gradient from a small random batch, which is much
cheaper and whose noise even helps to escape poor regions of the loss landscape.

**Backpropagation** computes all these gradients efficiently with the chain rule, reusing intermediate
results from the output layer backwards. Define the error signal of layer $l$ as
$\boldsymbol{\delta}^{(l)} = \partial \mathcal{L} / \partial \mathbf{z}^{(l)}$. Then:

1. Output layer: for a sigmoid (or softmax) output with cross-entropy loss the derivative simplifies
   remarkably to $\boldsymbol{\delta}^{(L)} = \mathbf{p} - \mathbf{y}$ (averaged over the batch).
2. Parameter gradients of layer $l$:
   $$ \frac{\partial \mathcal{L}}{\partial W^{(l)}} = \boldsymbol{\delta}^{(l)} \big(\mathbf{a}^{(l-1)}\big)^\top, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}. $$
3. Propagate the error one layer down ($\odot$ is the element-wise product):
   $$ \boldsymbol{\delta}^{(l-1)} = \Big(\big(W^{(l)}\big)^\top \boldsymbol{\delta}^{(l)}\Big) \odot \sigma'\big(\mathbf{z}^{(l-1)}\big). $$

These three equations are exactly what `NumpyMLP.backward()` in `main_1_mlp_numpy.py` implements (with
row-vector conventions, so the transposes appear on the other side). The cost of the backward pass is of
the same order as the forward pass – this efficiency is what makes training networks with millions or
billions of parameters possible.

Hand-written gradients are easy to get wrong, so the script verifies them with a **gradient check**: for a few
randomly chosen weights and biases it compares the analytical derivative with the central finite difference
$\big(\mathcal{L}(w+\epsilon) - \mathcal{L}(w-\epsilon)\big)/(2\epsilon)$. A relative error of about $10^{-7}$ or less
means the implementation is correct. Frameworks such as PyTorch remove the need to derive gradients by hand:
*autograd* records the operations of the forward pass in a computational graph and applies the chain rule
automatically when `loss.backward()` is called.

### Initialisation
The initial weights must break the symmetry between neurons (all-zero weights would make all neurons in a
layer identical forever) and keep the scale of the activations and gradients roughly constant across
layers. If the weights are too small the signal shrinks layer by layer (vanishing gradients); too large and it
explodes. Two standard choices, with $n_{\text{in}}$ and $n_{\text{out}}$ the fan-in and fan-out of the layer:

- **Xavier/Glorot** (tanh, sigmoid): $\mathrm{Var}(w) = 2/(n_{\text{in}} + n_{\text{out}})$,
- **He/Kaiming** (ReLU, which zeroes half of its inputs): $\mathrm{Var}(w) = 2/n_{\text{in}}$.

`main_1_mlp_numpy.py` trains the same network with He, Xavier and deliberately tiny ($\mathrm{std}=0.01$)
weights. With tiny weights the loss is stuck at $\log 2 \approx 0.693$ (random guessing) for dozens of epochs
before it starts to fall – a small-scale version of the vanishing-signal problem. PyTorch's `nn.Linear`
uses a scaled uniform initialisation by default; `main_2_mlp_pytorch.py` shows how to apply He/Xavier
explicitly via `nn.init`.

### Optimisers
Plain SGD uses the same step size for every parameter. Improvements used in practice:

- **Momentum** keeps a running average of past gradients, $\mathbf{v} \leftarrow \beta \mathbf{v} + \nabla\mathcal{L}$,
  $\theta \leftarrow \theta - \eta \mathbf{v}$, which damps oscillations and accelerates along consistent directions.
- **RMSprop** divides each parameter's step by a running root-mean-square of its gradients (adaptive per-parameter
  learning rates).
- **Adam** combines both (first and second moment estimates with bias correction) and is the robust default.
  **AdamW** decouples weight decay from the adaptive gradient scaling and is the standard for transformers.

`main_3_optimisers_regularisation.py` trains identical networks (same initial weights and batches) with the
four optimisers and plots the training loss on a log scale. Learning-rate *schedules* (step decay, cosine
annealing, reduce-on-plateau, warm-up) further improve convergence; `main_2_mlp_pytorch.py` uses
`ReduceLROnPlateau`.

### Regularisation and early stopping
A network with many parameters can memorise a small training set and generalise poorly (overfitting): the
training loss keeps decreasing while the validation loss increases. Remedies:

- **Weight decay / L2 regularisation**: add $\frac{\lambda}{2}\lVert\theta\rVert^2$ to the loss (or, in AdamW, shrink
  the weights directly by $\eta\lambda\theta$ each step), which favours smoother functions.
- **Dropout**: during training every hidden unit is set to zero with probability $p$ (and the rest scaled by
  $1/(1-p)$), so the network cannot rely on co-adapted units; it behaves like an ensemble of thinned networks.
  At evaluation time dropout is switched off – this is why `model.train()` and `model.eval()` matter.
- **Early stopping**: monitor the validation loss and stop when it has not improved for a number of epochs
  (*patience*), then restore the best weights. Implemented in `p06_neural_networks/utils/torch_helpers.py`.
- More data and data augmentation (chapter on CNNs), batch normalisation, smaller models.

The second experiment in `main_3_optimisers_regularisation.py` trains a 3×256 network on only 80 noisy
points. Without regularisation it fits every training point (train accuracy 100 %) with a very irregular
decision boundary, and its validation loss climbs steadily; dropout and weight decay keep the boundary
smooth and the validation loss low, and the combination generalises best.

### Walkthrough of the PyTorch script
`main_2_mlp_pytorch.py` follows the standard recipe:

1. **Data**: split into train/validation/test (60/20/20), standardise with training statistics only, wrap in
   `TensorDataset`s and `DataLoader`s (shuffled mini-batches for training).
2. **Model**: the `MLP` class builds `[Linear → activation → Dropout] × n` followed by a final `Linear`; the
   activation, width, depth and dropout rate are command-line options (`--activation tanh`, `--dropout 0`...).
   It returns logits: `nn.CrossEntropyLoss` combines log-softmax and negative log-likelihood in one
   numerically stable operation.
3. **Training loop** (`train_one_epoch` in the shared utils): forward pass, loss, `zero_grad()`, `backward()`,
   `step()`; then `evaluate()` on the validation set under `torch.no_grad()`.
4. **Early stopping** restores the weights of the best epoch; the test set is used once, at the very end.
5. **Saving**: `torch.save(model.state_dict(), ...)` stores the weights; to reload, construct the same model
   and call `model.load_state_dict(torch.load(path))`.

The scripts use `get_device()` (CUDA → MPS → CPU) and `set_seed()` from the shared utils so the results are
reproducible. On CPU, `get_device()` also enables `torch.set_flush_denormal(True)`: once a network fits its
training data, some gradients become so small that they are stored as *denormal* (subnormal) floating-point
numbers, and arithmetic on them can be much slower on some CPUs when many values become subnormal. A more
common cause of "training on the CPU is suddenly slow" is thread oversubscription: on a shared machine, or with
several processes training at once, each PyTorch process starts as many threads as there are cores, and they
compete for them. Setting `OMP_NUM_THREADS=1` (or calling `torch.set_num_threads(n)`) usually fixes it.

**Things to try:** change `--activation` to `sigmoid` with 6 layers and watch training slow down; set
`--dropout 0 --weight-decay 0 --patience 1000` and compare the learning curves; vary the learning rate over
orders of magnitude for each optimiser.
