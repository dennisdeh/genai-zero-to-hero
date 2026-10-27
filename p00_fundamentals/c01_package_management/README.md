## Environment
- Python conda environment to use: none of its own - this chapter is about creating environments.
  `conda_env_gzh_base.yml` is an example of a fully pinned `conda env export` (Linux) of a minimal scientific
  environment (`gzh_base`: Python 3.13, NumPy, pandas, SciPy, scikit-learn).
- Docker compose setup: none needed
- Environment variables: none needed
- Prerequisite: conda, preferably installed with Miniforge (https://github.com/conda-forge/miniforge), which uses
  the conda-forge channel by default.

## How to run
There is no code to run; the commands are run in a terminal from the repository root.

1. Configure conda to use conda-forge only, with strict channel priority (once):
   ```bash
   conda config --add channels conda-forge
   conda config --remove channels defaults
   conda config --set channel_priority strict
   ```
2. Create the environment of a part or chapter from its file and activate it, e.g.:
   ```bash
   conda env create -f p05_ml_classic/conda_env_gzh_ml0.yml
   conda activate gzh_ml0
   ```
   The pinned example in this folder only solves on Linux (x86-64):
   ```bash
   conda env create -f p00_fundamentals/c01_package_management/conda_env_gzh_base.yml
   ```
3. After the environment file has changed, update the environment (and remove packages no longer listed):
   ```bash
   conda env update -n gzh_ml0 -f p05_ml_classic/conda_env_gzh_ml0.yml --prune
   ```
4. Export a portable file with only the explicitly requested packages (add pip packages by hand):
   ```bash
   conda env export --from-history > conda_env_gzh_new.yml
   ```

More commands are collected in `conda_commands.md`; the background is explained in `book_package_management.md`.

## Files
- `book_package_management.md`: why environments, conda vs pip/venv vs uv, channels, pinning and lock files,
  conventions of this repository
- `conda_commands.md`: cheat sheet of useful conda commands
- `conda_env_gzh_base.yml`: example of a fully pinned conda environment export
