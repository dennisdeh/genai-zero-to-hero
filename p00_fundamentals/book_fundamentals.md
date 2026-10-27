# Fundamentals
This part contains the basic prerequisite concepts and technical setup.

The chapters cover best practices for development projects, package and environment management with conda,
and serving a machine learning model through a web API with FastAPI. The FastAPI service built here is the
running example that is containerised and deployed in the next part (p01).

## What this part covers
Machine learning and GenAI projects fail far more often because of engineering problems - code that only runs on
one laptop, results that cannot be reproduced, a model that cannot be called by anybody else - than because of the
choice of algorithm. This part therefore collects the tools and habits that every later part relies on:

- **c00_best_practices**: general principles for ML/GenAI projects (reproducibility, configuration and secrets,
  data and model versioning, logging), followed by two sections:
  - *s00_development_projects*: software-engineering practices - project layout, git workflow, code quality
    (ruff, type hints, mypy), testing with pytest, logging, configuration, documentation, pre-commit hooks and
    continuous integration - with a small, fully tested example package;
  - *s01_acknowledgements*: how to reuse open source code correctly (licences and attribution).
- **c01_package_management**: why isolated environments matter, conda vs pip/venv vs uv, channels, pinned vs
  unpinned environment files and lock files, and the conventions used for the environment files in this repository.
- **c02_fastapi**: wrapping a trained model in a validated, tested and documented web API - the first step from a
  script to a service.

## Setting up to work with the repository
The examples are designed to run on a normal laptop (Linux, macOS or Windows with WSL2). Most of them run offline
on a CPU in a minute or two; the exceptions are clearly marked in the chapter READMEs.

### Prerequisites
- **git**, to clone the repository and to follow the version-control practices of chapter c00.
- **conda**, preferably via the [Miniforge](https://github.com/conda-forge/miniforge) installer, which is configured
  to use the community `conda-forge` channel by default (see c01_package_management).
- **Docker** with docker compose (optional), for the services some chapters depend on (part p01 explains Docker).
- An **NVIDIA GPU** (optional), only needed for the LLM fine-tuning chapters and helpful for neural networks and local
  LLMs (see the GPU notes below).

### Clone the repository and create an environment
```bash
git clone https://github.com/dennisdeh/genai-zero-to-hero.git
cd genai-zero-to-hero
# every part (or chapter) has its own environment file, e.g. for classical machine learning:
conda env create -f p05_ml_classic/conda_env_gzh_ml0.yml
conda activate gzh_ml0
```
All environments are prefixed with `gzh_` and their files with `conda_env_gzh_`, so they are easy to find
(`conda env list`) and to remove again. Each chapter README names the environment it needs in its
"Environment" section. The environment files list the direct dependencies without version pins; chapter c01
explains this choice and how to pin versions when exact reproducibility is needed.

### Running the examples
Every script is run **from the repository root** as a module:
```bash
python -m p05_ml_classic.c00_linear_models.main_1_linear_regression
```
Running with `python -m package.module` puts the repository root on the Python path, so that the absolute imports
used throughout the code (e.g. `from p00_fundamentals.c02_fastapi.utils.config import Settings`) work no matter
which chapter a script lives in. Alternatively, set `PYTHONPATH=.` and run the file directly. The principal files
are called `main*.py`; numbered files (`main_1_...`, `main_2_...`) are meant to be run in order. Helpers live in
`utils/` subfolders, and tests in `test_*.py` files are run with `python -m pytest <folder>`.

Generated artefacts - trained models, plots, downloaded data, MLflow runs - are written to an `outputs/` folder
next to the script that creates them. These folders are ignored by git and can be deleted at any time.

### Configuration and secrets
Settings that differ between machines (ports, model names, the LLM provider) and secrets (API keys) are read from
environment variables, for example `LLM_TO_USE`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL` or `OPENAI_API_KEY` in parts
p07 and p08. Locally they are conveniently kept in a `.env` file in the repository root, which is loaded with
`python-dotenv` and is listed in `.gitignore`: **never commit a `.env` file or paste keys into code**. Chapter c00
explains the pattern in more detail.

### Services with Docker
Some chapters need a running service, started with docker compose from the chapter folder:
- a local LLM server ([Ollama](https://ollama.com/)) for parts p07 and p08, defined in
  `p07_llms/c01_running_llms/s01_ollama/docker-compose.yml` (it listens on host port 7011 and reserves an NVIDIA
  GPU; remove the `deploy` block to run it on a CPU, with slower responses);
- a vector database (Qdrant) for the RAG and agent chapters, defined in the `docker-compose.yml` files of
  `p07_llms/c04_rag_systems` and `p08_agents/c00_langchain_langgraph`.

The chapter READMEs list the exact commands, ports and environment variables.

### GPU notes
- Check that the driver works with `nvidia-smi`, and from Python with
  `python -c "import torch; print(torch.cuda.is_available())"`.
- The PyTorch wheels on PyPI ship with their own CUDA runtime libraries, so only a recent NVIDIA **driver** has to
  be installed on the machine - not the full CUDA toolkit. Apple Silicon Macs can use the `mps` backend instead
  of CUDA for many (not all) PyTorch operations.
- For Docker containers, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
  makes the GPU available inside the container.
- The examples choose the device automatically (`"cuda" if torch.cuda.is_available() else "cpu"`) and keep the
  default problem sizes small, so everything except the fine-tuning chapters of p07 also runs on a CPU.

### Building the book
The `book_*.md` files of all folders are concatenated (sorted by path) into a PDF by `build_book_pdf.sh` in the
repository root; see the root README for the required packages (pandoc and a LaTeX distribution).

## Mathematical prerequisites
The book is practical, but some mathematics is needed to understand why the methods work: **linear algebra**
(vectors, matrices, matrix products, eigenvalues), **calculus** (derivatives, gradients, the chain rule behind
backpropagation), and **probability and statistics** (distributions, expectation, likelihood, Bayes' rule,
hypothesis tests). The theory is introduced where it is needed - for example the normal equation and gradient
descent in p05, or attention in p07. Readers who want a systematic refresher can use the freely available books
*Mathematics for Machine Learning* by M. P. Deisenroth, A. A. Faisal and C. S. Ong (Cambridge University Press,
2020) and Part I ("Applied Math and Machine Learning Basics") of *Deep Learning* by I. Goodfellow, Y. Bengio and
A. Courville (MIT Press, 2016).
