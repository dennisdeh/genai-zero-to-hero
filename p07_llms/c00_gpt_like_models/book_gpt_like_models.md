## GPT-like Models

This chapter explores the design and implementation of GPT-like models, focusing on their architecture, training techniques, and applications. 
We will delve into the key components of these models, including the transformer architecture, attention mechanisms, and the role of pre-training and fine-tuning. Additionally, we will discuss the challenges and limitations of GPT-like models and explore potential future directions for research and development.

In this chapter we build two GPT-like models from scratch in PyTorch: a *tiny* word-level model that trains in
about a minute on a laptop (section `s00_tinygpt`), and a GPT-2-sized model based on Andrej Karpathy's minGPT that
is pretrained on Simple English Wikipedia and fine-tuned on financial questions and answers (section
`s01_minigpt`). Both follow the general workflow described in the introduction to this part: data preparation,
tokenisation, pretraining, fine-tuning and generation.

### The decoder-only transformer

GPT ("Generative Pre-trained Transformer") models are **decoder-only transformers** (Vaswani et al. 2017;
Radford et al. 2018, 2019). The model reads a sequence of token indices $x_1, \dots, x_T$ (at most the
*context length* or *block size* $T_{\max}$) and outputs, for every position $t$, a probability distribution over
the vocabulary for the *next* token $x_{t+1}$. The building blocks are the following.

#### Token and positional embeddings

Each token index is mapped to a learned vector of dimension $d$ (the embedding dimension, `n_embd` or
`embedding_dim`) by an embedding matrix $W_E \in \mathbb{R}^{V \times d}$, where $V$ is the vocabulary size.
Attention by itself is permutation-invariant, so the model also needs to know *where* a token is. GPT-1 and
GPT-2 use **learned absolute positional embeddings** $W_P \in \mathbb{R}^{T_{\max} \times d}$ which are simply
added: $h_t^{(0)} = W_E[x_t] + W_P[t]$. (The original transformer used fixed sinusoidal encodings; modern models
mostly use rotary position embeddings, RoPE, see chapter c05.) A consequence of learned absolute positions is
that the model cannot process sequences longer than $T_{\max}$; at generation time the context is therefore
cropped to the last $T_{\max}$ tokens.

#### Causal self-attention

In self-attention every position computes a *query*, a *key* and a *value* vector by linear projections of its
current representation, $Q = XW_Q$, $K = XW_K$, $V = XW_V$ with $X \in \mathbb{R}^{T \times d}$. The output is a
weighted average of the values, where the weights measure how well the query of position $t$ matches the keys of
all positions:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V .$$

Here $d_k$ is the dimension of the keys; dividing by $\sqrt{d_k}$ keeps the dot products at unit scale so that the
softmax does not saturate at initialisation. $M$ is the **causal mask**: $M_{ts} = 0$ for $s \le t$ and
$M_{ts} = -\infty$ for $s > t$, so that after the softmax a position can only attend to itself and to earlier
positions. The mask is what makes the model *autoregressive*: the prediction for position $t$ never sees the
token it has to predict, and one forward pass over a sequence of length $T$ yields $T$ training examples at once.
In code the mask is a lower-triangular matrix (`torch.tril`) and the masked entries are filled with `-inf`
before the softmax.

#### Multi-head attention

Instead of a single attention with $d_k = d$, the model runs $h$ attention **heads** in parallel, each with
its own projections of size $d_k = d/h$, concatenates their outputs and mixes them with an output projection
$W_O$:
$$\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h) W_O .$$
Different heads can specialise in different relations (the previous token, the subject of the sentence, a
matching bracket, ...), at the same cost as one large head.

#### MLP, residual connections and layer normalisation

Each **transformer block** consists of the attention sub-layer followed by a position-wise **MLP** (feed-forward
network) with a hidden layer of size $4d$ and a non-linearity (ReLU in the original transformer, GELU in GPT).
Attention moves information *between* positions; the MLP processes each position independently and holds a
large part of the model's parameters and "knowledge".

Both sub-layers are wrapped in **residual connections** and **layer normalisation**. GPT-2 uses the
*pre-norm* arrangement, which trains more stably for deep networks than the original *post-norm* arrangement:
$$x \leftarrow x + \mathrm{MHA}(\mathrm{LN}(x)), \qquad x \leftarrow x + \mathrm{MLP}(\mathrm{LN}(x)).$$
The residual stream lets gradients flow directly through many layers; layer normalisation normalises each
token vector to zero mean and unit variance (with a learned scale and shift). After the last block a final layer
norm is applied, and a linear **language-model head** $W_U \in \mathbb{R}^{d \times V}$ maps every position to
$V$ *logits*.

#### Weight tying

The input embedding $W_E$ maps tokens to vectors and the output head $W_U$ maps vectors back to tokens. Many
models (GPT-2 among them) **tie** these weights, $W_U = W_E^\top$, which saves $V \cdot d$ parameters (about 38
million of the 124 million parameters of GPT-2 small) and tends to improve sample efficiency because every token
embedding receives gradients both as input and as output (Press and Wolf 2017). Neither implementation in this
chapter ties the weights: minGPT keeps a separate `lm_head`, and when it loads the original GPT-2 weights
(`GPT.from_pretrained`) it simply copies the same matrix into both places. Tying would be a one-line change
(`self.lm_head.weight = self.transformer.wte.weight`).

### Training objective

Pretraining is self-supervised: the targets are the inputs shifted by one position. For a training sequence
the model minimises the average **cross-entropy** (negative log-likelihood) of the next token,
$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_{t+1} \mid x_1, \dots, x_t),$$
where $p_\theta$ is the softmax of the logits at position $t$. In PyTorch this is
`F.cross_entropy(logits.view(-1, V), targets.view(-1))`. The exponential of the loss is the **perplexity**: a
model with loss $\mathcal{L}$ is, on average, as uncertain as a uniform choice among $e^{\mathcal{L}}$ tokens. A
randomly initialised model has a loss of about $\log V$ (e.g. $\log 85 \approx 4.44$ for the tiny model below,
which indeed starts at a loss of about 4.5). Training uses AdamW with weight decay on the matrix weights only,
gradient clipping and, for larger models, learning-rate warm-up and decay.

**Fine-tuning** continues training the pretrained weights on a smaller, task-specific dataset, usually with a
much smaller learning rate so that the general knowledge from pretraining is not destroyed ("catastrophic
forgetting"). Full fine-tuning updates all weights; parameter-efficient methods such as LoRA are covered in
chapter c03.

### Generation and sampling

A trained model generates text **autoregressively**: feed the prompt, take the logits of the last position,
choose a next token, append it and repeat until a maximum length or an end-of-text token is reached. How the next
token is chosen matters a lot:

- **Greedy decoding** takes the most likely token (`argmax`). It is deterministic but tends to produce
  repetitive or short, generic text.
- **Sampling** draws the token from the distribution $\mathrm{softmax}(z / \tau)$, where $z$ are the logits and
  $\tau$ is the **temperature**. $\tau < 1$ sharpens the distribution (more conservative), $\tau > 1$
  flattens it (more diverse, more errors); $\tau \to 0$ recovers greedy decoding.
- **Top-k sampling** sets all logits outside the $k$ largest to $-\infty$ before the softmax, so that the long
  tail of unlikely tokens can never be chosen. **Top-p (nucleus) sampling** (Holtzman et al. 2019) instead keeps
  the smallest set of tokens whose cumulative probability exceeds $p$.

minGPT's `GPT.generate(idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None)` implements greedy
decoding, temperature and top-k; the scripts in `s01_minigpt` contain a manual sampling loop that additionally
stops at the end-of-text token.

### Section s00_tinygpt: a tiny word-level GPT

The first section (adapted from Aarohi Singla's "Build a Mini GPT Model From Scratch Using PyTorch", see the
section README) shrinks every component to the minimum so that the whole pipeline is visible in two files.

`transformer_blocks.py` contains the building blocks, written out explicitly:

- `SelfAttentionHead`: key, query and value projections without bias, a `tril` buffer as causal mask, the
  scaled scores `q @ k.transpose(-2, -1)`, `masked_fill(..., -inf)`, softmax and the weighted sum of the values.
  Note that the scores are divided by $\sqrt{C}$ with $C$ the *embedding* dimension rather than the head size
  $d_k$ used in the formula above (and in minGPT). This only changes the effective temperature of the attention
  and is harmless at this size, but it is a nice detail to spot.
- `MultiHeadAttention`: `num_heads` heads of size `embedding_dim // num_heads` whose outputs are concatenated
  and projected back to `embedding_dim`.
- `FeedForward`: the MLP with a hidden layer of size `4 * n_embd` and ReLU.
- `Block`: a pre-norm transformer block, `x = x + sa(ln1(x))` followed by `x = x + ffwd(ln2(x))`.

`main.py` contains the data, the model and the training loop:

1. **Data and tokenisation**: 50 short sentences about European history, each terminated with an `<END>`
   token, are joined into one text. The tokeniser is the simplest possible one, a *word-level* tokeniser: every
   distinct word is a token (85 tokens in total), with the dictionaries `word2idx` and `idx2word` for encoding
   and decoding. The whole corpus becomes a tensor of 346 token indices.
2. **Batches**: `get_batch` picks 16 random positions and cuts out windows of `block_size = 6` tokens as input
   `x` and the same windows shifted by one token as target `y`.
3. **Model**: `TinyGPT` combines a token embedding, a learned position embedding, `n_layers = 2` blocks with
   `n_heads = 2` heads and `embedding_dim = 32`, a final `LayerNorm` and a linear head to the vocabulary. The
   `forward` method returns the logits and, if targets are given, the cross-entropy loss.
4. **Training**: 1500 steps of AdamW with learning rate $10^{-2}$; the loss drops from about 4.5 to about
   0.4-0.5. With only 346 tokens the model essentially memorises the corpus - which is the point: we can watch
   a transformer learn.
5. **Generation**: `generate` starts from the token `europe`, crops the context to the last `block_size`
   tokens, takes the **greedy** argmax of the last logits and stops at `<END>`. Because many training sentences
   end with "... europe <END>", greedy decoding frequently produces just `europe <END>`; changing the start word
   (e.g. `king` or `people`) or replacing the argmax by sampling (`torch.multinomial(F.softmax(logits, -1), 1)`)
   is an instructive exercise.

The script imports `transformer_blocks` as a top-level module and must therefore be run from inside the section
folder (see the README).

### Section s01_minigpt: pretraining and fine-tuning a GPT-2-sized model

The second section is a fork of Karpathy's **minGPT** (MIT licence, see `LICENSE` and the section README),
extended by the owner with a data pipeline, a trained BPE tokeniser, multi-GPU support, validation-loss based
early stopping and MLflow logging. It walks through the first three steps of the LLM workflow in three scripts.

#### The minGPT library (`mingpt/`)

- `model.py`: `CausalSelfAttention` computes the query, key and value projections for all heads with one linear
  layer (`c_attn`, output size $3d$), reshapes them to `(B, n_head, T, head_size)`, applies the scaled, causally
  masked softmax and the output projection `c_proj`, with dropout on the attention weights and on the residual
  path. `Block` is the pre-norm block with a GELU MLP. `GPT` assembles token embedding `wte`, position embedding
  `wpe`, dropout, `n_layer` blocks, the final `ln_f` and the `lm_head`. The configuration is selected by name -
  `gpt2` (12 layers, 12 heads, $d = 768$, 124M parameters with the GPT-2 vocabulary), `gpt-mini`, `gpt-micro`,
  `gpt-nano` and others. Weights are initialised from $\mathcal{N}(0, 0.02^2)$, and the residual projections
  with a standard deviation scaled by $1/\sqrt{2 n_{\text{layer}}}$ as in GPT-2. `configure_optimizers` puts the
  weights of linear layers in an AdamW group with weight decay and biases, layer-norm and embedding weights in a
  group without. `generate` implements greedy decoding, temperature and top-k sampling, and `from_pretrained`
  loads the original GPT-2 weights from Hugging Face.
- `trainer.py`: a model-agnostic training loop - an infinite random-sampling `DataLoader`, forward pass,
  `loss.mean()` (needed when `torch.nn.DataParallel` returns one loss per GPU), backward pass, gradient clipping
  and optimiser step, and callbacks on `on_batch_end`. The owner's additions are `estimate_loss` (average loss
  on random validation examples in `eval` mode), `batch_end_callback` (prints the training loss every 50 and
  the validation loss every 250 iterations) and **early stopping**: if a callback returns a validation loss that
  has not improved for `early_stopping_rounds` evaluations, training stops.
- `bpe.py` (OpenAI's GPT-2 byte-pair encoder) and `utils.py` (seeding, logging and the small `CfgNode`
  configuration class). `mingpt/tests/` contains a unit test checking that minGPT reproduces the Hugging Face
  GPT-2 logits.

#### Data collection (`data_collection/`) and tokenisation (`tokenisation/`)

`data_collection.py` downloads the datasets with the Hugging Face `datasets` library:

- `dc_wiki` loads the **Simple English Wikipedia** dump `wikimedia/wikipedia`, `20231101.simple` and cleans it in
  two passes: sentences are split with NLTK's `sent_tokenize`; very long sentences (tables, references, SVG
  code), sentences with paragraph breaks, numbers-only or code fragments are skipped; bracketed content and
  accents are removed and whitespace is normalised; a second pass drops sentences with suspicious punctuation and
  stops at date lists. It returns a dictionary title -> text (or list of sentences).
- `dc_finance_qna` loads the **finance-alpaca** dataset (`gbharti/finance-alpaca`), keeps the `instruction`
  and `output` columns as question/answer pairs and removes duplicates, empty and overly long entries.

`tokeniser.py` trains a **byte-level BPE tokeniser** with the Hugging Face `tokenizers` library on the cleaned
Wikipedia text: a vocabulary of 32,000 tokens, a byte-level pre-tokeniser (any input string can be encoded,
there are no unknown characters) and the special tokens `<|pad|>`, `<|startoftext|>`, `<|separation|>`,
`<|endoftext|>` and `<|unknown|>`. Byte-pair encoding starts from single bytes and repeatedly merges the most
frequent adjacent pair into a new token, so frequent words become single tokens while rare words are split into
sub-words. A tokeniser trained on the domain needs fewer tokens per sentence than a general one; alternatively the
scripts accept OpenAI's `tiktoken` `cl100k_base` encoding.

`data.py` turns text into PyTorch datasets:

- `data_preparation_wiki` splits every article into training and validation text (either the first 90% of the
  sentences for training and the rest for validation, or random sentences), encodes each article as
  `<|startoftext|> ... <|endoftext|>`, concatenates everything into one long token stream and wraps it in
  `TextDataset`, which returns overlapping windows of `block_size` tokens and the same window shifted by one as
  target - the standard way of *packing* a corpus for pretraining.
- `data_preparation_finance` formats each pair as
  `<|startoftext|> question <|separation|> answer <|endoftext|>` padded with `<|pad|>` to exactly `block_size`
  tokens (answers are truncated if necessary) and wraps the samples in `QnADataset`. The separator teaches the
  model where the question ends and the answer begins - a minimal form of the *chat templates* used by
  instruction-tuned models. Note that the loss is computed on all positions, including the question and the
  padding; masking these targets (setting them to `-1`, which minGPT's `cross_entropy(..., ignore_index=-1)`
  ignores) would focus the training signal on the answers.

#### The three steps

1. `main_step1_data_preparation.py` loads and cleans the full Wikipedia dump, trains the custom tokeniser
   `custom_wiki_bpe_32k`, saves it to `trained_tokenisers/` and prints the encoding of an example sentence and the
   ids of the special tokens.
2. `main_step2_pretraining.py` pretrains a `gpt2`-sized model (12 layers, 12 heads, $d = 768$, block size 512,
   vocabulary 32,000) from scratch on the Wikipedia token stream: AdamW with learning rate $10^{-4}$, weight decay
   0.01, gradient clipping at 1.0, batch size 32 (16 on a single GPU), at most 30,000 iterations with early
   stopping after 50 evaluations without improvement. The training and validation losses and the hyperparameters
   are logged to an MLflow server at `http://localhost:5000`, the weights are saved to `trained_models/` as
   `model_<timestamp>.bin`, and a short sample is generated. With `training = False` (the default) the script
   instead loads the model file named in `load_model` and generates a continuation of "He then went to the beach"
   with plain multinomial sampling until `<|endoftext|>`. According to the script docstring, good performance is
   reached after about 15k iterations - which requires a GPU.
3. `main_step3_finetuning_full.py` loads the pretrained weights (`load_pt_model`) and continues training *all*
   weights on the finance Q&A samples with a much smaller learning rate ($10^{-6}$), at most 5,000 iterations and
   early stopping after 10 evaluations without improvement, saving `model_ft_<timestamp>.bin`. With
   `training = False` it loads `load_ft_model` and samples from it; to see the effect of fine-tuning, use a prompt
   in the training format, e.g. `"How do I make money?<|separation|>"`.

The file names of the models in `load_model`, `load_pt_model` and `load_ft_model` refer to the owner's training
runs; `trained_models/` and `trained_tokenisers/` are gitignored, so you have to run the steps yourself (or adjust
the names) before inference works.

#### Projects and notebooks

The `projects/` folder contains minGPT's original small demos: `adder` trains a `gpt-nano` model to add n-digit
numbers (the digits of the result are reversed, which makes the carry easier to learn), and `chargpt` trains a
character-level `gpt-mini` language model on any text file, e.g. Tiny Shakespeare. `demo.ipynb` trains a
model to sort sequences and `generate.ipynb` loads the pretrained GPT-2 weights and generates text. These
scripts use minGPT's original import style (`from mingpt.model import GPT`), so `s01_minigpt` has to be on the
Python path (see the chapter README).

### Limitations and outlook

The models in this chapter show *all* the ingredients of an LLM, but at a scale that is many orders of magnitude
smaller than production models: GPT-3 has 175B parameters and was trained on about 300B tokens, current open
models on more than 10T tokens. Scale (Kaplan et al. 2020; Hoffmann et al. 2022 for the compute-optimal ratio of
parameters to tokens) is what turns next-token prediction into the broad capabilities of modern LLMs. Since GPT-2
the architecture has been refined in many details - RoPE instead of learned positions, RMSNorm instead of
LayerNorm, SwiGLU MLPs, grouped-query attention and FlashAttention for efficiency, mixture-of-experts layers -
but the decoder-only skeleton above is unchanged. After pretraining, models are made useful as assistants by
instruction tuning and preference optimisation (RLHF, DPO); efficient fine-tuning is the topic of chapter c03. Known limitations remain:
hallucinations (the objective rewards plausible, not true, text), a fixed knowledge cut-off (addressed by
retrieval, chapter c04), limited context length (chapter c05) and the large cost of training and inference.
