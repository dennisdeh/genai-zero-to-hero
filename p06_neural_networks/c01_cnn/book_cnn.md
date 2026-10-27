## Convolutional Neural Networks

An MLP treats an image as a flat vector: every pixel is connected to every hidden unit, the spatial
neighbourhood structure is ignored, and an object shifted by a few pixels looks completely different.
Convolutional neural networks (CNNs) build two assumptions about images into the architecture:

- **Locality**: useful features (edges, strokes, textures) depend on small neighbourhoods of pixels.
- **Translation equivariance**: a feature detector useful in one place is useful everywhere, so the same
  weights are *shared* across all positions.

These priors drastically reduce the number of parameters and make CNNs the classic architecture for vision
(LeNet 1998, AlexNet 2012, VGG, ResNet), audio spectrograms and other grid-like data.

| Script | What it shows |
|---|---|
| `main_1_cnn_classifier.py` | a VGG-style CNN with batch norm, data augmentation, training, confusion matrix, checkpoints |
| `main_2_visualise_cnn.py` | learned filters, feature maps captured with forward hooks, saliency maps |

The default dataset is scikit-learn's bundled 8×8 handwritten digits (1,797 images), so everything runs
offline in under a minute. With `--dataset fashion_mnist` (28×28 clothing images) or `--dataset cifar10`
(32×32 colour photos) the same code downloads a larger benchmark via torchvision; `--subset` limits the
number of training images for a quick CPU run.

### The convolution layer
A 2D convolution slides a small kernel $K$ of size $k \times k$ over the image $X$ and computes a weighted
sum at every position (strictly speaking a cross-correlation, but the name stuck):
$$ Y_{i,j} = b + \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} K_{u,v}\, X_{i+u,\, j+v}. $$
With $C_{\text{in}}$ input channels (e.g. RGB) and $C_{\text{out}}$ output channels, each output channel has its own
$C_{\text{in}} \times k \times k$ kernel and sums over all input channels:
$$ Y_{c,i,j} = b_c + \sum_{c'=1}^{C_{\text{in}}} \sum_{u,v} K_{c,c',u,v}\, X_{c',\, i+u,\, j+v}. $$
A `Conv2d(64, 128, kernel_size=3)` therefore has $128 \cdot 64 \cdot 9 + 128 \approx 74{,}000$ parameters,
independent of the image size – a fully connected layer between two $64\times 32\times 32$ and
$128 \times 32 \times 32$ tensors would need $\approx 8.6 \cdot 10^9$.

The output size along each spatial dimension is
$$ n_{\text{out}} = \left\lfloor \frac{n_{\text{in}} + 2p - k}{s} \right\rfloor + 1 $$
for padding $p$ and stride $s$; with $k=3$, $p=1$, $s=1$ ("same" padding, used in the scripts) the resolution is
preserved. The output channels are called *feature maps*: each one shows where in the image its filter
responded strongly.

**Receptive field.** Every layer sees only a $k\times k$ window of the previous one, but stacking layers grows
the region of the *input* that influences a unit: two $3\times3$ convolutions see $5\times5$ pixels, three see
$7\times7$ – with fewer parameters and more non-linearities than one large kernel (the VGG insight).
Pooling and strides grow the receptive field faster.

### Pooling, batch normalisation and the classifier head
- **Max pooling** (`MaxPool2d(2)`) keeps the maximum of every $2\times2$ window, halving the resolution. It
  adds some invariance to small shifts and reduces computation; channels are usually doubled when the
  resolution is halved.
- **Batch normalisation** normalises each channel over the mini-batch,
  $\hat{x} = (x - \mu_{\mathcal{B}})/\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}$, followed by a learnable scale and shift
  $\gamma \hat{x} + \beta$. It stabilises and accelerates training and allows higher learning rates. During
  evaluation the running averages of $\mu$ and $\sigma^2$ collected during training are used – another reason to
  call `model.eval()`. (A convolution followed by batch norm needs no bias: $\beta$ plays that role.)
- **Global average pooling** (`AdaptiveAvgPool2d(1)`) averages each final feature map to a single number, so the
  network works for any input resolution and needs only a small linear layer on top.

The model in `main_1_cnn_classifier.py` is a small VGG-style network:
$$ \underbrace{[\text{Conv-BN-ReLU} \times 2 \to \text{MaxPool}]}_{32 \text{ channels}} \to \underbrace{[\ldots]}_{64} \to \underbrace{[\text{Conv-BN-ReLU} \times 2]}_{128} \to \text{GAP} \to \text{Dropout} \to \text{Linear}. $$
On 8×8 digits the resolution goes 8 → 4 → 2 → 1. Modern architectures add *residual connections*
($\mathbf{y} = \mathbf{x} + F(\mathbf{x})$, ResNet), which make networks with hundreds of layers trainable.

### Data augmentation
The cheapest way to get more training data is to transform the existing images in ways that do not change
the label: small rotations, shifts and scalings (`v2.RandomAffine`), horizontal flips for natural images
(not for digits – a mirrored 2 is not a 2), crops, colour jitter. The transforms are applied *on the fly* in the
`Dataset.__getitem__`, so the network sees a different variant of every image in every epoch. Augmentation is
used for training only; validation and test images are just normalised. Run with `--no-augment` to compare.

### Training, evaluation and checkpoints
The training loop is the one from the MLP chapter (`train_one_epoch`, `evaluate`, `EarlyStopping` from
`p06_neural_networks/utils/torch_helpers.py`) plus:

- **AdamW** with a **cosine annealing** learning-rate schedule and **label smoothing** (the target distribution
  puts 0.95 on the true class instead of 1, which discourages over-confident predictions; it is also why the
  loss does not go to zero).
- **Evaluation** on the held-out test set with scikit-learn's `classification_report` (precision, recall, F1 per
  class), a **confusion matrix** (which digits are confused with which) and a gallery of misclassified images –
  looking at errors is often the most informative step in improving a model.
- **Checkpoints**: `save_checkpoint()` stores a dictionary with the model `state_dict`, the optimiser state, the
  epoch, the hyper-parameters and the normalisation statistics. `load_checkpoint()` rebuilds the model from
  the stored config and loads the weights with `torch.load(..., weights_only=True)`, which refuses to unpickle
  arbitrary Python objects (never load untrusted pickles without it). The script evaluates the *reloaded* model
  to prove the round trip works. Saving the optimiser state allows training to be resumed exactly.

Expect about 99 % test accuracy on the digits within a few dozen epochs (a logistic regression reaches about
96 %) and about 85 % on Fashion-MNIST with a 10,000-image subset and only 5 epochs (shirts are the hardest class –
they are confused with T-shirts, pullovers and coats). More data and epochs improve this further; CIFAR-10 is
considerably harder and benefits from the full training set, more epochs and a GPU.

### Looking inside the network
`main_2_visualise_cnn.py` loads the checkpoint (training one first if necessary) and produces three views:

1. **First-layer filters**: the $3\times3$ kernels plotted as images (red positive, blue negative weights). On
   natural images, first-layer filters typically become edge and colour-blob detectors; on tiny digits they are
   less interpretable but still show oriented stroke detectors.
2. **Feature maps** of a test image after every ReLU, recorded with *forward hooks*:
   `module.register_forward_hook(fn)` calls `fn(module, input, output)` after the module's forward pass – a
   general technique for inspecting or modifying intermediate activations without changing the model code.
   The printed statistics show how sparse the ReLU activations are (around half are exactly zero).
3. **Saliency maps** (Simonyan et al. 2013): the gradient of the predicted class score with respect to the input
   pixels, $|\partial s_c / \partial X|$. It highlights which pixels would change the decision most and is the
   simplest member of the family of attribution methods (Grad-CAM, integrated gradients, SHAP).

**Things to try:** train without batch norm (replace it by `nn.Identity()`) and compare the learning curves;
remove augmentation; train on Fashion-MNIST and inspect the confusion matrix (shirts vs. T-shirts vs. pullovers);
replace the max-pooling by strided convolutions.
