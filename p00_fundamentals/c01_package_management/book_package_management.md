## Package and Environment Management
Every example in this book depends on third-party packages: NumPy, pandas, scikit-learn, PyTorch, LangChain and
many more. Managing these dependencies well is the foundation of code that runs on more than one machine. This
chapter explains why isolated environments are needed, compares the main tools (conda, pip with venv, and uv),
and describes the conventions used for the environment files of this repository. A cheat sheet of useful conda
commands is found in `conda_commands.md` in this chapter's folder.

### Why environments?
Installing everything into one global Python installation quickly leads to problems:

- **Conflicting requirements**: project A needs `numpy<2`, project B needs `numpy>=2`; both cannot be installed in
  the same place. Deep-learning stacks are particularly sensitive - PyTorch, CUDA libraries, `transformers` and
  quantisation libraries must come in compatible versions.
- **Different Python versions**: an older project may need Python 3.10 while a new one uses 3.13.
- **Breaking the system**: on Linux and macOS the operating system's own tools use the system Python; installing
  or upgrading packages there can break them (recent distributions therefore refuse `pip install` into the system
  Python, see PEP 668).
- **Reproducibility**: "it works on my machine" is only useful if the machine's software can be described and
  recreated elsewhere - by colleagues, on a server, in a Docker image.

An **environment** is an isolated directory containing a Python interpreter and a set of packages. Each project
(here: each part or chapter) gets its own environment, described by a text file that is committed to git.

### The tools: conda, pip/venv and uv
**pip and venv.** `pip` installs packages from the Python Package Index ([PyPI](https://pypi.org/)), mostly as
pre-built *wheels*. `venv` (part of the standard library) creates lightweight environments that reuse an existing
Python interpreter:
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn
pip freeze > requirements.txt      # exact versions of everything installed
pip install -r requirements.txt    # recreate elsewhere
```
pip only manages Python packages; the Python version itself and non-Python libraries (compilers, CUDA, GDAL) must
come from elsewhere.

**conda.** [conda](https://docs.conda.io/) is a language-agnostic package and environment manager. Its packages
contain compiled binaries of anything - Python itself, C/C++ and Fortran libraries, CUDA, R, command-line tools - and
its solver resolves the dependencies of the whole environment at once. This makes it well suited for the scientific
stack, where Python packages depend on large native libraries (BLAS/LAPACK, MKL, OpenMP, HDF5). Since version 23.10,
conda uses the fast `libmamba` solver by default; `mamba` is a drop-in reimplementation of the conda command line.
```bash
conda create --name gzh_demo python=3.13 numpy pandas
conda activate gzh_demo
conda install scikit-learn
```

**uv.** [uv](https://docs.astral.sh/uv/) is a recent, very fast replacement for pip, venv, pip-tools and
pyenv, written in Rust by the authors of ruff. It can install Python versions, manages a project's dependencies in
`pyproject.toml` and writes a cross-platform lock file `uv.lock`:
```bash
uv init my-project && cd my-project   # creates pyproject.toml
uv add numpy scikit-learn             # adds dependencies, updates .venv and uv.lock
uv run python main.py                 # runs inside the project environment
uv sync                               # recreate the environment exactly from uv.lock
uv pip install -r requirements.txt    # pip-compatible interface
```
Poetry, PDM and Hatch are other project managers built around `pyproject.toml`; pixi brings lock files and
project-based workflows to conda packages.

| | pip + venv | conda | uv |
|---|---|---|---|
| package source | PyPI | conda channels (e.g. conda-forge) | PyPI |
| non-Python dependencies | no (bundled in wheels at best) | yes | no |
| installs Python itself | no | yes | yes |
| lock files | via pip-tools (`pip-compile`) | via conda-lock or explicit spec files | built in (`uv.lock`) |
| speed | moderate | moderate (libmamba solver) | very fast |
| typical use | simple projects, Docker images | data science, mixed native stacks | modern Python projects |

There is no single right choice. This book uses **conda** environments for working locally, because they handle
the Python version and native libraries uniformly across operating systems, and plain **pip** requirement files
inside the Docker images of part p01, where the base image already provides Python.

#### Mixing conda and pip
Many packages, especially new GenAI libraries, are only available on PyPI, so conda environments often contain pip
packages too. Rules of thumb:

1. install as much as possible with conda first, then the remaining packages with pip;
2. list pip packages in the `pip:` subsection of the environment file, so that both are installed in the right
   order;
3. avoid running `conda install` after `pip install` in the same environment - conda does not know about the
   pip-installed packages and may overwrite their dependencies; if in doubt, recreate the environment from the file.

PyTorch is a typical example: its official wheels on PyPI already contain the CUDA runtime libraries, so in this
book it is installed with pip and only the NVIDIA driver must be present on the machine.

### Channels and conda-forge
conda packages are downloaded from **channels**. The most important ones are:

- **conda-forge**: a community-maintained channel with the largest collection of packages (tens of thousands), built
  and updated automatically from public recipes. It is free to use for everybody.
- **defaults** (the Anaconda repository): packages built by Anaconda Inc. Its use is subject to Anaconda's terms of
  service, which require a paid licence for larger organisations.

Mixing packages from different channels can lead to incompatible binaries. The recommendation is to use
conda-forge exclusively with **strict channel priority** (see the end of `conda_commands.md`):
```bash
conda config --add channels conda-forge
conda config --remove channels defaults
conda config --set channel_priority strict
```
The [Miniforge](https://github.com/conda-forge/miniforge) installer is configured like this out of the box and is
the recommended way to install conda for this book.

### Environment files: unpinned, pinned and lock files
An environment is described by a YAML file:
```yaml
name: gzh_ml0
channels:
  - conda-forge
dependencies:
  - python=3.13
  - pip
  - numpy
  - scikit-learn
  - pip:
      - some-pypi-only-package
```
created with `conda env create -f conda_env_gzh_ml0.yml` and updated after changes to the file with
`conda env update -n gzh_ml0 -f conda_env_gzh_ml0.yml --prune` (`--prune` removes packages that are no longer
listed). Such files can be written at three levels of precision:

1. **Unpinned (intent)**: only the direct dependencies, without versions (or with loose bounds such as
   `numpy>=2`). The solver picks the newest compatible versions for the current platform. The file is short,
   readable, portable across Linux, macOS and Windows, and stays current - but two people creating the environment
   a year apart get different versions.
2. **Fully pinned export**: `conda env export > file.yml` lists *every* installed package, including indirect
   dependencies, with exact version and build string (e.g. `numpy=2.4.0=py313hf6604e3_0`), plus a `prefix:` line
   with the local installation path. `conda_env_gzh_base.yml` in this folder is such an export. It reproduces the
   environment exactly - but only on the same platform: build strings and low-level packages such as
   `_libgcc_mutex` or `libgcc` exist only for Linux, so the file typically fails to solve on macOS or Windows. The
   `prefix:` line is machine-specific and can be deleted.
3. **Lock files**: a machine-generated, fully pinned description of the environment for one or more platforms,
   kept *in addition* to the human-written intent file. Examples are `conda-lock` (`conda-lock -f environment.yml
   -p linux-64 -p osx-arm64` writes `conda-lock.yml`), `conda list --explicit > spec-file.txt` (exact package URLs
   for one platform, recreated with `conda create -n name --file spec-file.txt`), `pip-compile` from pip-tools
   (`requirements.in` to a pinned `requirements.txt`), and `uv.lock`.

To produce a **portable** file from an existing environment, export only the packages that were explicitly requested:
```bash
conda env export --from-history > conda_env_gzh_new.yml
```
This lists the specs exactly as they were typed in `conda create`/`conda install` (e.g. `python=3.13`, `numpy`),
without the indirect dependencies and build strings. Packages installed with pip are *not* included and must be
added to the `pip:` section by hand; `conda env export --no-builds` is a middle ground that keeps all versions but
drops the build strings.

Version specifiers differ slightly between the tools: in conda, `numpy=2.1` means any `2.1.*` and `numpy==2.1.3` an
exact version; in pip, `numpy==2.1.3` is exact, `numpy~=2.1` means `>=2.1,<3` and `numpy>=2,<3` a range.

#### Which level to use?
- **Learning material, libraries and exploratory work**: unpinned or loosely bounded files. This keeps the
  examples current; when a new release breaks something, the error is visible and can be fixed.
- **Applications, deployed models and published results**: an intent file *plus* a lock file (or a pinned
  requirements file inside a Docker image), updated deliberately and tested in CI. A model serialised with pickle
  or joblib in particular must be loaded with the same library versions it was saved with.

### Conventions in this repository
- Environment names are prefixed with `gzh_` (for "genai-zero-to-hero"), so they are easy to recognise in
  `conda env list` and to remove again.
- Environment files are named `conda_env_gzh_<name>.yml` and live in the folder they belong to: one per part where
  the chapters share their dependencies (e.g. `p05_ml_classic/conda_env_gzh_ml0.yml` for `gzh_ml0`), or one per
  chapter where the dependencies differ a lot (e.g. `gzh_llm0` to `gzh_llm7` in p07, `gzh_api0` for the FastAPI
  chapter, `gzh_dev0` for the best-practices chapter). The README of each chapter names the file to use.
- New files are written **unpinned** with `channels: [conda-forge]`, so that they work across operating systems and
  pick up current versions. Some older files (`gzh_base` in this chapter and several environments of p07 and p08)
  are full Linux exports: they reproduce the original environment exactly on Linux, while on macOS or Windows an
  environment is best recreated from its main packages (Python, the libraries the code imports and the `pip:`
  section) without versions and build strings.
- New environment files are created with `conda env export --from-history`, the pip packages are added by hand,
  and the `prefix:` line is removed.
- Docker images (part p01) use `requirements.txt` files with lower bounds and explain how to pin them for
  production.

A typical workflow for a new chapter:
```bash
conda create --name gzh_new python=3.13 --channel conda-forge --override-channels
conda activate gzh_new
conda install numpy pandas scikit-learn       # conda packages first
pip install some-pypi-only-package            # then pip packages
conda env export --from-history > conda_env_gzh_new.yml
# edit the file: add the pip packages under "- pip:", remove the "prefix:" line
```
To use an environment in Jupyter, register it as a kernel with
`python -m ipykernel install --user --name gzh_new` (requires `ipykernel` in the environment).

### Reproducibility beyond the environment file
An environment file pins the Python packages, but not the operating system, system libraries, GPU driver or
hardware. The levels of reproducibility, from weakest to strongest, are:

1. a list of direct dependencies (unpinned environment file, `requirements.in`);
2. a lock file with exact versions of all dependencies, per platform;
3. a container image (part p01) that also fixes the operating system and system libraries; its digest identifies it
   exactly;
4. the same image on the same hardware with fixed seeds and deterministic algorithms (see chapter c00).

Record the versions with every result (the example in `c00_best_practices` stores the Python and NumPy versions
and the git commit in its run summary; MLflow does this automatically), and recreate environments from their files
regularly - an environment that can only be reproduced by its original owner is a liability.
