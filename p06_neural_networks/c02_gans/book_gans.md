## Generative Adversarial Networks

The networks in the previous chapters are *discriminative*: they learn $p(y \mid \mathbf{x})$, a label given an
input. *Generative* models learn the data distribution $p_{\text{data}}(\mathbf{x})$ itself, so that new samples –
images, sounds, molecules – can be drawn from it. Generative adversarial networks (GANs, Goodfellow et al. 2014)
do this without ever writing down a likelihood: a **generator** network $G$ turns random noise
$\mathbf{z} \sim p_z = \mathcal{N}(0, I)$ into samples $G(\mathbf{z})$, and a **discriminator** network $D$ tries to tell
real samples from generated ones. The two are trained against each other; the generator improves by learning
to fool the discriminator.

| Script | What it shows |
|---|---|
| `main_1_gan_2d.py` | a GAN on a 2D ring of 8 Gaussians: minimax vs. non-saturating loss, training tricks, mode collapse |
| `main_2_dcgan_digits.py` | a DCGAN generating 16×16 handwritten digits, sample quality/diversity check, latent interpolation |

Both run offline on a CPU in about a minute.

### The minimax game
$D(\mathbf{x}) \in (0,1)$ is the probability that $\mathbf{x}$ is real. The original GAN objective is a two-player
minimax game:
$$ \min_G \max_D \; V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}\big[\log D(\mathbf{x})\big] + \mathbb{E}_{\mathbf{z} \sim p_z}\big[\log\big(1 - D(G(\mathbf{z}))\big)\big]. $$
For the discriminator this is just the binary cross-entropy of a classifier with labels real $=1$, fake $=0$.
For a fixed generator with sample distribution $p_g$ the optimal discriminator is
$$ D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}, $$
and plugging it back in gives
$$ V(D^*, G) = -\log 4 + 2\, \mathrm{JS}\big(p_{\text{data}} \,\Vert\, p_g\big), $$
where JS is the Jensen–Shannon divergence. The generator therefore minimises a divergence between the real and
the generated distribution, with the global optimum $p_g = p_{\text{data}}$ (where $D^* = 1/2$ everywhere).

### Non-saturating loss
In practice the generator's term $\log(1 - D(G(\mathbf{z})))$ is a poor training signal. Early in training the
generator is bad, the discriminator rejects its samples confidently, $D(G(\mathbf{z})) \approx 0$, and the
function $\log(1 - D)$ is flat there – the gradient *saturates* and the generator barely learns. Goodfellow et al.
therefore proposed the **non-saturating** loss: instead of minimising $\log(1 - D(G(\mathbf{z})))$, the generator
maximises $\log D(G(\mathbf{z}))$, i.e. it minimises
$$ \mathcal{L}_G = -\mathbb{E}_{\mathbf{z}}\big[\log D(G(\mathbf{z}))\big], $$
which has the same fixed point but strong gradients exactly when the generator is losing. In code this is
simply binary cross-entropy of the fake samples against the label "real":
`F.binary_cross_entropy_with_logits(D(G(z)), ones)`. `generator_loss()` in `main_1_gan_2d.py` implements both
variants; the non-saturating loss is what virtually every implementation uses.

### The training loop
Each iteration alternates two gradient steps:

1. **Discriminator step**: sample a real batch and a fake batch `G(z).detach()` (detaching stops gradients from
   flowing into $G$), compute BCE with labels 1 and 0, update $D$.
2. **Generator step**: sample new fakes, pass them through $D$ *without* detaching, compute $\mathcal{L}_G$ and update
   only $G$'s parameters (the discriminator receives gradients too, but its optimiser is not stepped).

Unlike ordinary training, there is no single loss that goes down: the losses of $D$ and $G$ oscillate as each
player adapts to the other. A discriminator loss around $2\log 2 \approx 1.39$ (it cannot distinguish real from
fake) is a sign of a balanced game, not of failure. Progress must be judged by looking at samples or by
dedicated metrics.

### Training difficulties and tricks
GAN training is a search for a saddle point, not a minimum, and is notoriously unstable. Typical failure modes:

- **Mode collapse**: the generator maps many $\mathbf{z}$ to the same few outputs that currently fool $D$. When $D$
  learns to reject them, $G$ jumps to another mode ("mode hopping") instead of covering all of them.
- **Vanishing gradients**: a too-strong discriminator gives no useful signal (mitigated by the non-saturating loss).
- **Oscillation / divergence**: the players keep chasing each other without converging.

Widely used tricks (many from the DCGAN paper and Salimans et al. 2016):

- Adam with a lower momentum, $\beta_1 = 0.5$, and moderate learning rates ($2\cdot10^{-4}$ for images).
- **One-sided label smoothing**: real labels $0.9$ instead of $1$, which keeps the discriminator from becoming
  over-confident. (Smoothing the fake labels would shift the optimum, hence "one-sided".)
- LeakyReLU in the discriminator (gradients also for negative inputs), batch normalisation, careful initialisation.
- Balancing the players (more discriminator steps per generator step, `--d-steps` in the script).
- Better objectives: Wasserstein GAN with gradient penalty (WGAN-GP), hinge loss, spectral normalisation; and for
  diversity: minibatch discrimination, unrolled GANs.

### Experiment 1: a ring of Gaussians
`main_1_gan_2d.py` uses the classic toy problem from the mode-collapse literature: a mixture of 8 Gaussians
arranged on a circle. Being two-dimensional, the generated distribution can be plotted directly and two simple
metrics can be computed: the number of modes that receive samples and the fraction of "high-quality" samples
(within three standard deviations of a mode centre). Both $G$ and $D$ are MLPs with three hidden layers.
Three set-ups are trained with the same seed and architecture:

- **non_saturating** and **minimax** – both with the tricks above ($\beta_1=0.5$, learning rate $10^{-3}$, label
  smoothing). Both cover all 8 modes after about 1,000 steps; the generated distribution first spreads around the
  ring and then concentrates on the modes. With these tricks even the minimax loss works on this small problem
  (try `--d-steps 5` to make the discriminator stronger and compare how the two losses cope).
- **no_tricks** – a five times larger learning rate, the default $\beta_1 = 0.9$ and no label smoothing. The
  generator collapses onto a single mode at a time and hops between modes, often producing samples far away
  from any mode in between: textbook mode collapse and instability.

The snapshots (`outputs/gan_2d_snapshots.png`) and the coverage curves (`outputs/gan_2d_mode_coverage.png`)
make the difference obvious at a glance.

### Experiment 2: DCGAN on handwritten digits
Radford et al. (2015) established architectural guidelines that made convolutional GANs train reliably
("DCGAN"), and `main_2_dcgan_digits.py` follows them on the bundled scikit-learn digits (upsampled to 16×16 and
scaled to $[-1,1]$):

- **Generator**: the noise vector is treated as a $1\times1$ image with `z_dim` channels and upsampled by
  *transposed convolutions* ($1 \to 2 \to 4 \to 8 \to 16$ pixels) with batch norm and ReLU, ending in `tanh`.
  A transposed convolution with kernel 4, stride 2 and padding 1 exactly doubles the resolution.
- **Discriminator**: the mirror image – strided convolutions instead of pooling, LeakyReLU(0.2), batch norm except
  on the input layer, and a single logit as output.
- Weights initialised from $\mathcal{N}(0, 0.02)$, Adam with learning rate $2\cdot10^{-4}$ and $\beta_1 = 0.5$.

Image grids generated from a *fixed* noise batch are saved at several epochs, so the progress from blobs to
recognisable digits can be followed. Remember to switch the generator to `eval()` mode for sampling, so batch
norm uses its running statistics.

**How good are the samples?** GAN evaluation is hard; for natural images the standard metrics are the Inception
Score and the Fréchet Inception Distance (FID), which compare statistics of a pre-trained image network on
real and generated samples. The script uses the same idea at toy scale: a logistic regression trained on real
digits labels 2,000 generated images. If all ten digits appear about equally often (entropy of the class
histogram close to $\log 10 \approx 2.30$) there is no mode collapse; the mean classifier confidence measures
how recognisable the samples are. After 80 epochs the histogram is close to uniform and the samples are
clearly readable digits.

Finally, **latent-space interpolation**: decoding points on the straight line between two noise vectors
$\mathbf{z}_0$ and $\mathbf{z}_1$, $G((1-\alpha)\mathbf{z}_0 + \alpha \mathbf{z}_1)$, gives a smooth morph from one digit into
another – evidence that the generator has learned a continuous representation rather than memorised training
images.

### Where GANs stand today
GANs dominated image synthesis from about 2015 to 2021 (progressive GANs, StyleGAN, BigGAN) and are still used where
fast, single-pass generation matters (super-resolution, image-to-image translation, real-time applications) and
as adversarial losses inside other models. For high-fidelity, diverse image generation, diffusion models have
largely taken over because they train stably and do not suffer from mode collapse, at the price of slower,
iterative sampling. The adversarial idea itself – learning a loss function with a second network – remains a
powerful tool.

**Things to try:** set `--z-dim 2` in the DCGAN and plot a grid over the 2D latent space; train the 2D GAN with
`--d-steps 5 --configs minimax`; change `CONFIGS["no_tricks"]` one setting at a time to find out which trick
matters most.
