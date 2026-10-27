### Software Development
This section covers best practices for software development projects, including version control, testing, and documentation.

Data scientists often start in notebooks, and notebooks are excellent for exploration. But as soon as code is
reused, shared, scheduled or deployed, it needs the same discipline as any other software: a clear structure,
version control, automated checks and tests. The practices below are standard in software engineering; the
emphasis here is on what is special about ML code.

The section comes with a small example package in `s00_development_projects`:

- `utils/scaling.py`: feature standardisation (z-scores) written to the standards described below - type hints,
  docstrings, input validation, logging and no hidden global state;
- `utils/config.py`: configuration from environment variables in a validated, immutable dataclass;
- `utils/reproducibility.py`: seeding, a timing context manager and run metadata (versions, git commit);
- `main_best_practices.py`: a script that ties it together - it shows why feature scaling matters for gradient
  descent, while demonstrating logging, configuration, seeding, timing and traceable outputs;
- `test_scaling.py`, `test_config.py`: pytest tests demonstrating fixtures, parametrisation, `tmp_path`,
  `monkeypatch`, `caplog`, `pytest.approx` and property-style tests;
- `example_pyproject.toml`, `example_pre-commit-config.yaml`: example configuration files for ruff, black, pytest,
  mypy and pre-commit. They are stored here as examples and are *not* active for the repository.

#### Project layout
A predictable layout lets anybody (including your future self) find things quickly. A typical ML project looks like
this:

```
my-ml-project/
|-- README.md               # what it is, how to install, how to run
|-- pyproject.toml          # metadata, dependencies, tool configuration
|-- environment.yml         # conda environment (or requirements.txt / uv.lock)
|-- .gitignore              # data/, outputs/, models/, .env, caches
|-- .env.example            # names of the required environment variables, dummy values
|-- .pre-commit-config.yaml
|-- configs/                # experiment configurations (YAML/TOML)
|-- data/                   # local data, NOT in git (raw/ is read-only, processed/ is generated)
|-- notebooks/              # exploration; reusable code moves into src/
|-- src/my_project/         # the importable package: data, features, models, evaluation, serving
|-- scripts/                # entry points: train.py, evaluate.py
`-- tests/                  # pytest tests, mirroring the package structure
```

The key ideas are: code that is used more than once lives in an importable package (not in notebooks); entry points
are thin scripts that parse the configuration and call package functions; data, models and outputs are generated,
not committed. Templates such as *Cookiecutter Data Science* provide a ready-made version of this structure.

This repository follows a variant adapted to a book: each chapter folder is a package with `main*.py` entry
points, a `utils/` subpackage with helpers and `test_*.py` files next to the code; scripts are run from the
repository root with `python -m`, and generated files go to `outputs/` folders that git ignores.

#### Version control with git
Git records every change of the code base, makes it possible to work in parallel and to go back to any earlier
state. A few habits make the history useful:

- **Small, focused commits** with messages that explain *why*: a short summary line in the imperative
  ("Add stratified split to training data"), a blank line, and details if needed.
- **Branches**: `main` always works. New work happens on short-lived feature branches
  (`git switch -c feature/stratified-split`) that are merged back through a **pull request** (PR).
- **Pull requests and code review**: a PR describes the change and how it was tested; automated checks (CI, see
  below) run on it and at least one colleague reviews it. Reviews catch bugs, but just as importantly spread
  knowledge about the code base.
- **What not to commit**: data, model weights, outputs, caches, virtual environments and secrets. List them in
  `.gitignore` *before* the first commit; a file that has been committed stays in the history even after it is
  deleted. For example:

```
# .gitignore (excerpt)
data/
outputs/
models/
*.ckpt
*.safetensors
.env
__pycache__/
.ipynb_checkpoints/
mlruns/
```

- **Notebooks**: their outputs (images, data) make diffs unreadable and can leak data; strip outputs before
  committing (e.g. with `nbstripout`) or keep notebooks out of the main code path.
- **Tags and releases** mark versions that were deployed or used for a publication (`git tag v1.2.0`), so the
  exact code can always be recovered.

The daily cycle is: `git switch main && git pull`, `git switch -c <branch>`, edit and test, `git add -p` (review
each change), `git commit`, `git push -u origin <branch>`, open a PR, address review comments, merge.

#### Code quality: style, linting and type checking
Consistent code is easier to read and review. Instead of discussing style, let tools enforce it:

- **Formatter**: rewrites code to a consistent style. `black` was the de facto standard; `ruff format` is a much
  faster, black-compatible replacement. Use one of them, with the same line length everywhere.
- **Linter**: finds bugs and bad patterns without running the code - unused imports and variables, undefined
  names, mutable default arguments, `zip()` of sequences of different length, outdated syntax. `ruff` implements
  the rules of flake8, isort, pyupgrade, bugbear and many other tools and fixes many of them automatically
  (`ruff check --fix`).
- **Type hints** document what a function expects and returns (`def standardise(x: np.ndarray, params:
  ScalerParams) -> np.ndarray`). They are not enforced at runtime, but editors use them for autocompletion and
  **mypy** checks them statically, finding errors such as passing a string where a number is expected, or a
  possible `None` that is not handled. Start with `disallow_untyped_defs` for new code and tighten gradually;
  libraries without type information are handled with `ignore_missing_imports`.

All three are configured in `pyproject.toml` (see `example_pyproject.toml`). The example code of this section passes
`ruff check`, `ruff format --check`, `black --check` and `mypy` with that configuration.

#### Testing with pytest
Tests are code that checks code. They catch regressions when something is changed, document how functions are
meant to be used, and make refactoring safe. [pytest](https://docs.pytest.org/) is the standard framework: tests
are plain functions whose names start with `test_`, in files named `test_*.py`, using plain `assert` statements.
pytest finds them automatically and, on failure, shows the values involved.

```bash
python -m pytest p00_fundamentals/c00_best_practices -v      # verbose: one line per test
python -m pytest -k "split" -x                               # only tests matching "split", stop at first failure
```

The most important features, all used in `test_scaling.py` and `test_config.py`:

- **Fixtures** provide test data and resources. A function decorated with `@pytest.fixture` is called by pytest and
  its return value is passed to every test that has a parameter of the same name; fixtures can use other fixtures,
  can be shared across a module or session (`scope="module"`), and can clean up after the test (with `yield`).
  Built-in fixtures include `tmp_path` (a fresh temporary directory), `monkeypatch` (temporarily set environment
  variables or attributes; changes are undone after the test) and `caplog` (captured log messages).
- **Parametrisation** runs one test over many inputs: `@pytest.mark.parametrize("seed", range(5))`. Stacking two
  decorators runs the full grid (in the example, 5 seeds $\times$ 3 shapes = 15 test cases from one function).
- **Floating-point comparisons** must use a tolerance: `assert value == pytest.approx(0.3)` instead of
  `== 0.3`, or `np.testing.assert_allclose(a, b, rtol=..., atol=...)` for arrays.
- **Exceptions**: `with pytest.raises(ValueError, match="2-dimensional"):` checks that invalid input fails loudly
  with a helpful message.

##### Testing ML code
ML code is harder to test because it involves randomness, floating-point arithmetic and outputs without a single
"correct" answer. Useful strategies:

- **Determinism**: seed every random number generator in a test (a fixture returning
  `np.random.default_rng(0)`), so that a failing test fails every time - flaky tests are worse than no tests.
- **Small fixtures**: tests use tiny synthetic datasets (tens of rows) that run in milliseconds, not the real
  data. A handful of hand-computed examples (`test_fit_known_values`) pin down the exact behaviour.
- **Shape and type tests**: the cheapest tests catch a surprising number of bugs - output shapes, dtypes, no NaNs,
  probabilities in $[0, 1]$ that sum to one.
- **Property-based tests**: instead of expected numbers, check properties that hold for *any* input. For
  standardisation: the output has mean 0 and standard deviation 1 for every column; `inverse(transform(x))`
  returns `x`; and changing the units of a feature ($x \mapsto ax + b$ with $a > 0$) does not change the z-scores.
  The example runs these over several seeds and shapes with `parametrize`; the
  [Hypothesis](https://hypothesis.readthedocs.io/) library (not used here) generates such inputs automatically,
  including nasty edge cases.
- **Leakage tests**: preprocessing fitted on the training data must not be refitted on test data
  (`test_test_data_uses_training_statistics`).
- **Behavioural tests of models**: invariance (a model's prediction should not change when an irrelevant feature
  changes), directional expectations (a larger flat should not get a lower predicted price), and minimum
  performance on a small, fixed dataset ("the model beats the baseline by at least X"), with tolerances.
- **Smoke tests**: run the whole pipeline end to end on a tiny configuration (`test_main_runs_end_to_end` uses
  200 samples and writes to `tmp_path`) to catch integration errors.

The test coverage (the fraction of lines executed by the tests) is measured with the `pytest-cov` plugin
(`pytest --cov=my_project`); high coverage is no proof of correctness, but untested code is likely to be broken.

#### Logging instead of print
`print` is fine for a quick experiment, but `logging` is better for anything that lives longer:

- each message has a **level**; switching from INFO to DEBUG makes the program more verbose without changing code;
- each message has a **timestamp** and the **name of the logger** (by convention the module, `logging.getLogger(__name__)`),
  so it is clear where it came from;
- the **destination** (console, file, a log aggregation service, JSON for machines) is configured once, at the
  entry point of the application, with `logging.basicConfig(...)` or handlers. Library modules only create loggers
  and never configure handlers.

Use lazy formatting, `logger.info("Loaded %d rows", n)`, so the string is only built if the message is emitted, and
`logger.exception(...)` inside an `except` block to include the traceback. `main_best_practices.py` logs to the
console and to `outputs/run.log` at the same time. Never log secrets or personal data.

#### Configuration
Hard-coded paths and hyperparameters scattered through the code make it impossible to run the same code in another
environment or to know which settings produced a result. Instead:

- collect all settings in one place, e.g. a frozen dataclass (as in `utils/config.py`), a pydantic `BaseSettings`
  class, or a YAML/TOML file for experiment configurations (tools such as Hydra manage hierarchies of them);
- read environment variables for anything that differs between machines, with defaults for harmless settings;
- **validate at start-up** (`__post_init__` in the example raises a `ValueError` for an invalid test fraction),
  so that a typo fails immediately and not after hours of training;
- pass the configuration explicitly to the functions that need it, which keeps them testable;
- save the configuration together with every result.

In the example, `GZH_SEED=7 GZH_LOG_LEVEL=DEBUG python -m ...main_best_practices` changes the seed and the
verbosity without touching the code, and `monkeypatch.setenv` does the same in the tests.

#### Documentation
- **README.md**: what the project does, how to install it, how to run it, how to run the tests - the first (and
  often only) file people read. Every chapter of this book has one.
- **Docstrings** for every public module, class and function: what it does, the arguments, the return value and
  the exceptions raised. The examples use the Google style (`Args:`, `Returns:`, `Raises:`), which tools such as
  Sphinx or MkDocs can turn into a documentation website.
- **Comments explain why, not what**: `# population std (ddof=0), as in scikit-learn` is useful;
  `# compute the std` is not.
- **Type hints** are documentation that is checked by a tool.
- **Decision records** and a changelog for larger projects explain why things are the way they are.

#### Pre-commit hooks
Git can run scripts automatically before each commit. The [pre-commit](https://pre-commit.com/) framework manages
such hooks from a `.pre-commit-config.yaml` file: it installs every tool in an isolated environment at a pinned
version and runs it on the staged files. `example_pre-commit-config.yaml` configures:

- file hygiene from `pre-commit-hooks`: `end-of-file-fixer`, `trailing-whitespace`, `check-yaml`, `check-toml`,
  `check-merge-conflict`, `check-added-large-files` (keeps data and models out of git) and `detect-private-key`;
- `ruff-check --fix` and `ruff-format` from `ruff-pre-commit` (black and mypy are included as commented-out
  alternatives).

To use it in a project: copy it to `.pre-commit-config.yaml` in the repository root, then run
`pre-commit install` once; from then on every `git commit` runs the checks. `pre-commit run --all-files` runs them
on the whole code base, and `pre-commit autoupdate` bumps the pinned versions. If a hook modifies files, the commit
is aborted so the changes can be reviewed and added.

#### Continuous integration
Pre-commit hooks run on the developer's machine and can be skipped. **Continuous integration** (CI) runs the same
checks - plus the full test suite - on a server for every push and pull request, so nothing broken is merged. With
GitHub Actions, a minimal workflow in `.github/workflows/ci.yml` looks like this:

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.13"
      - run: pip install numpy pytest ruff mypy
      - run: ruff check . && ruff format --check .
      - run: mypy .
      - run: python -m pytest -q
```

For ML projects, CI typically runs the fast unit tests on every push and slower jobs (training on a sample,
evaluation against a baseline, building a Docker image) on merges or nightly. Continuous deployment of the
FastAPI model service is covered in part p01.

#### The example: why standardisation matters
`main_best_practices.py` generates a synthetic housing dataset whose four features live on very different scales
(floor area around 140 square metres, number of rooms around 4, distance to the centre around 15 000 m, age around 60 years),
splits it, fits the scaler on the training set only, and trains a linear regression with full-batch gradient
descent for 500 steps - once on the raw and once on the standardised features. The step size is $1/L$, where $L$ is
the largest eigenvalue of the Hessian $\frac{2}{n}X^\top X$, which guarantees convergence; the *speed* of
convergence is governed by the condition number $\kappa$ of $X^\top X$, the ratio of its largest to its smallest
eigenvalue. Typical output:

| features | $\kappa(X^\top X)$ | test RMSE after 500 steps | exact least squares |
|---|---|---|---|
| raw | about $4.6\times 10^{9}$ | 317 | 19.6 |
| standardised | about 1.1 | 19.6 | 19.6 |

With raw features the loss surface is an extremely long, narrow valley: the step size has to be small enough for the
steepest direction (the distance feature), so progress along the other directions is negligible and the script logs
a warning. After standardisation the valley is almost round and gradient descent reaches the exact solution. The
same reasoning explains why neural networks are trained on normalised inputs (part p06).

Along the way the script logs its settings, run metadata (Python and NumPy versions, platform, git commit) and the
duration of each step, verifies that the same seed reproduces the same data, and writes `run.log`,
`scaler_params.json` and `run_summary.json` to its `outputs/` folder - everything needed to understand and
reproduce the run later.
