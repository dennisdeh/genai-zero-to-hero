## Best Practices
This chapter covers the best practices for developing and deploying GenAI solutions. 
It includes guidelines for code organisation, documentation, testing, and deployment.

ML and GenAI projects are software projects, but with two extra sources of change: the **data** and the
**model**. A conventional program behaves the same as long as its code does not change; a model can silently
become worse because the data changed, because a random seed changed, because a library was upgraded, or because
the hosted LLM behind an API was updated. Good engineering practice is what turns the inevitable experimentation
into results that can be trusted, reproduced and handed over to other people.

The chapter is split into two sections:

- **Software development** (`s00_development_projects`): the practices of professional software development
  applied to ML code - project layout, git workflow, code quality tooling, testing with pytest (including how to
  test code that involves randomness and floating-point numbers), logging, configuration, documentation,
  pre-commit hooks and continuous integration. It comes with a small example package with tests, a script and
  example configuration files for ruff, pytest, mypy and pre-commit.
- **Acknowledgements** (`s01_acknowledgements`): how to reuse open source code legally and fairly - checking
  licences, attributing authors and declaring modifications.

The remainder of this introduction lists principles that apply to every ML/GenAI project and that are used
throughout the book.

### General principles for ML/GenAI projects
#### Reproducibility
A result that cannot be reproduced cannot be debugged, compared or trusted. A run is reproducible when the same
**code**, **data**, **configuration**, **environment** and **randomness** give the same result. Each of these is
pinned down by a different tool:

| ingredient | how to pin it down |
|---|---|
| code | git commit hash, recorded with every result |
| data | immutable, versioned datasets (see below) |
| configuration | config files or environment variables, saved next to the results |
| environment | conda/pip environment files and lock files (chapter c01), Docker images (part p01) |
| randomness | explicit seeds |

A practical habit: every run writes a small summary (configuration, git commit, library versions, metrics) next
to its artefacts. The example script of section s00 does exactly this, and MLflow automates it (see below).

#### Seeding and determinism
Random numbers appear everywhere: train/test splits, weight initialisation, data shuffling, dropout, sampling from
an LLM. Fixing the seeds makes runs repeatable:

- Python: `random.seed(seed)`; NumPy: prefer an explicit generator `rng = np.random.default_rng(seed)` that is
  passed to the functions that need it, over the global `np.random.seed(seed)`;
- scikit-learn: pass `random_state=seed` to estimators and splitters;
- PyTorch: `torch.manual_seed(seed)` (seeds CPU and GPU), and `torch.use_deterministic_algorithms(True)` if
  bit-identical results on a GPU are required (some GPU kernels are non-deterministic by default and this option
  can make training slower);
- LLMs: `temperature=0` makes generation (mostly) greedy, and some APIs accept a `seed` parameter, but hosted models
  give no hard guarantee of identical outputs.

Seeds make runs *repeatable*, not *robust*: a conclusion that only holds for one seed is not a conclusion. Report
results over several seeds (or cross-validation folds) when comparing models.

#### Configuration and secrets
Code should not change when it moves from a laptop to a server. Everything that differs between environments -
paths, hyperparameters, ports, model names, credentials - belongs in **configuration**, following the
[twelve-factor app](https://12factor.net/config) principle of storing configuration in environment variables:

- read the configuration in one place at start-up, validate it, and pass it explicitly to the code that needs it
  (e.g. a frozen dataclass or a pydantic `BaseSettings` class);
- give harmless settings sensible defaults, but never give secrets a default value;
- keep local values in a `.env` file (loaded with `python-dotenv`), add `.env` to `.gitignore` and commit a
  `.env.example` with the variable *names* and dummy values instead;
- never log, print or commit secrets. If a key has been committed, removing it from the latest commit is not
  enough - it remains in the git history, so **revoke and rotate the key**.
  Pre-commit hooks such as `detect-private-key` and secret scanners on the git host catch many such mistakes.

#### Data and model versioning
Git is made for text files of moderate size, not for gigabytes of data or model weights. Large files are kept out of
git (`.gitignore` the `outputs/`, `data/` and model folders) and versioned with dedicated tools:

- **immutable raw data**: never modify the raw data in place; derive processed datasets with code, so they can be
  recreated;
- **data versioning** with tools such as [DVC](https://dvc.org/) (git-like versioning of data stored in cloud
  buckets) or [Git LFS](https://git-lfs.com/), or simply versioned, read-only folders/buckets with a hash of the
  content;
- **model versioning** with a model registry (e.g. the MLflow model registry), which stores every model version
  together with the run, data and parameters that produced it;
- for LLM applications, the **prompts** and the name and version of the model they were developed for are
  versioned like code.

#### Experiment tracking
During development dozens of experiments are run. Recording them in spreadsheets or file names does not scale.
Experiment trackers record the parameters, metrics, artefacts, code version and environment of every run
automatically and make runs comparable in a web interface. This book uses MLflow; see
`p05_ml_classic/c03_mlops` for experiment tracking, the model registry and drift monitoring, and
`p07_llms/c07_validation` for the evaluation of LLM applications.

#### Logging and observability
Use the `logging` module instead of `print`: log records carry a level (DEBUG, INFO, WARNING, ERROR), a timestamp
and the module they come from, and the application decides at one place where they go (console, file, a log
aggregation service) and how verbose they are. In production, log the information needed to debug a problem -
inputs (without personal data), model version, latency, errors - and monitor the model's inputs and outputs
for drift. Section s00 shows how to set this up.

#### Further principles
- **Start simple**: a baseline (a linear model, a heuristic, a prompt without fine-tuning) gives a reference that
  every more complex solution must beat.
- **Automate the boring parts**: formatting, linting, tests and builds run automatically (pre-commit, CI), so that
  reviews can focus on the content.
- **Make small, reviewed changes**: small commits and pull requests are easier to review, test and revert.
- **Document decisions**, not only code: why a dataset, a model or a threshold was chosen (this repository records its
  design decisions in `implementation_options.md`).
- **Respect licences and data protection**: check the licence of code, data and pretrained models before use
  (section s01), and keep personal data out of logs, prompts sent to third parties and git repositories.
