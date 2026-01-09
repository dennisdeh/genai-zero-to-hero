# Large Language Models
## Introduction
In this part we will look at Large Language Models (LLMs).

### Creating an LLM
Training an LLM is a multi-stage process that combines 
large-scale data processing, optimisation, and distributed systems (i.e. cloud infrastructure).

The general workflow can be broadly divided into the following steps:
1. Data collection and preparation 
2. Model pretraining
3. Fine-tuning
4. Validation
5. Deployment
6. Monitoring 

The process begins by defining the project aim and constraints. 
These decisions directly influence architectural choices and dataset composition.
The dataset must be prepared and filtered to remove unwanted content and duplicates.
Fit-for-purpose analyses regarding bias, ethics, safety, and legal compliance are crucial to ensure the model's utility and acceptability.
The dataset is then tokenised and packed into fixed-length sequences to prepare it for use in the training process.

Most modern LLMs are (pre-)trained for autoregressive next-token prediction, which requires large amounts of training data.
As such, this is an unsupervised machine-learning task that just requires a large amount of unlabeled data of high quality.
This is by far the most expensive part of the process and requires a significant investment.
The point of the pre-training stage is to learn general linguistic and semantic structure that can be leveraged by downstream tasks.

It is of course possible to take model pre-trained by a third-party as a starting point to circumvent this step entirely.
There are several well-performing open-source models available for this purpose as we will explore later.
The fine-tuning step adjusts the weights of the model to better suit the specific task at hand.
This is a supervised learning task and requires a large amount of labeled data.

This can speed up the development process significantly, and lead to faster deployment, modulo appropriate validation and monitoring.

More details for each of the follow below.


#### Data collection and preparation
Training data is collected at very large scale from diverse sources such as web text, books, academic papers, and code repositories. 
The goal is to maximise coverage of languages, domains, and writing styles while maintaining legal and ethical compliance. 
Raw data is then filtered to remove duplicates, low-quality content, and undesired material such as spam, corrupted text, or personally identifiable information. 
Language identification and heuristic quality scoring are typically applied to ensure consistency and usefulness of the corpus.

Once the text corpus is cleaned, it is transformed into a numerical representation through tokenisation. 
Subword tokenisation schemes such as Byte Pair Encoding (BPE) or unigram language models are commonly used to balance vocabulary size against sequence length. 
The resulting token streams are segmented into fixed-length sequences, packed to reduce padding overhead, shuffled, and sharded to support distributed training. 
Data is stored in formats optimised for high-throughput streaming to GPUs.

#### Model pretraining
LLMs are almost exclusively based on transformer architectures, most often in a decoder-only configuration. 
The architecture is defined by parameters such as the number of layers, hidden dimensionality, attention heads, and maximum context length. 
Additional design choices include positional encoding schemes and attention optimisations for efficiency. 
Model parameters are initialised using carefully scaled random distributions to ensure stable gradient flow during early training or, 
in some cases, loaded from an existing pretrained checkpoint.

Pretraining is the core computational phase, during which the model learns general linguistic and semantic structure by minimising 
the negative log-likelihood of the next token over massive datasets. 

Training is performed on large clusters using combinations of data, model, and pipeline parallelism. 
Considering LLMs not produced by large corporations like OpenAI, frameworks like `PyTorch` can be applied, with optimisers such as `AdamW`
paired with learning rate warmup and decay schedules to stabilise convergence. 
Throughout pretraining, engineers monitor loss curves, gradient statistics, and throughput, while periodically 
saving checkpoints to guard against failures.

To maintain stability and generalisation, techniques such as gradient clipping, weight decay, 
and mixed-precision loss scaling are applied. 
The model is regularly evaluated on a validation dataset to track perplexity and detect overfitting. 
Lightweight downstream tasks are sometimes used as probes to assess emerging capabilities and 
data contamination issues before training completes.


#### Fine-tuning
After pretraining, the model is adapted to more specific behaviours through supervised fine-tuning on curated instruction–response datasets. 
This stage teaches the model how to follow prompts, format outputs, and adhere to task conventions.
Further alignment is often achieved through preference-based optimisation methods such as reinforcement 
learning from human feedback or direct preference optimisation, which changes the model toward producing relevant and desired outputs.

#### Validation
Before deployment, extensive testing is conducted to evaluate robustness, bias, hallucination tendencies, 
and susceptibility to adversarial prompting. 
Red-teaming exercises and automated safety benchmarks are used to identify failure modes. 
Findings from this phase often lead to additional fine-tuning or data filtering.

#### Deployment
To make inference practical, the trained model may undergo compression through quantisation, 
pruning, or distillation.

The optimised model is then deployed behind scalable inference infrastructure with batching, caching, and monitoring. 
Runtime metrics such as latency, throughput, and error rates should be continuously tracked.

#### Monitoring
After deployment, real-world usage data and feedback are analysed to detect distribution shifts, misuse, or degraded performance. 
This information feeds back into later data collection, fine-tuning, and retraining cycles, enabling continuous 
improvement of the model over time.


