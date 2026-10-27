## Environment
- Python conda environment to use: `conda_env_gzh_dev0.yml` (environment `gzh_dev0` with numpy, pytest and the
  code-quality tools ruff, black, mypy and pre-commit), e.g.
  `conda env create -f p00_fundamentals/c00_best_practices/conda_env_gzh_dev0.yml && conda activate gzh_dev0`
- Docker compose setup: none needed
- Environment variables (all optional, read by `s00_development_projects/utils/config.py`):
  - `GZH_SEED`: random seed (default: `42`)
  - `GZH_N_SAMPLES`: number of synthetic samples (default: `2000`)
  - `GZH_TEST_FRACTION`: fraction of samples in the test set (default: `0.25`)
  - `GZH_LOG_LEVEL`: logging level, e.g. `DEBUG`, `INFO`, `WARNING` (default: `INFO`)
  - `GZH_SAVE_OUTPUTS`: write artefacts to the outputs folder, `true`/`false` (default: `true`)
  - `GZH_OUTPUT_DIR`: folder for the artefacts (default: `s00_development_projects/outputs`)

  They can also be put in a `.env` file in `s00_development_projects/` (loaded if `python-dotenv` is installed;
  never commit it).

## How to run
All commands are run from the repository root; everything finishes in a few seconds on a CPU.

1. Run the example script (logging, configuration, seeding, timing; raw vs standardised features in gradient descent):
   ```bash
   python -m p00_fundamentals.c00_best_practices.s00_development_projects.main_best_practices
   # with a different configuration:
   GZH_SEED=7 GZH_LOG_LEVEL=DEBUG python -m p00_fundamentals.c00_best_practices.s00_development_projects.main_best_practices
   ```
   The log is printed to the console; `run.log`, `scaler_params.json` and `run_summary.json` are written to
   `p00_fundamentals/c00_best_practices/s00_development_projects/outputs/` (git-ignored).
2. Run the tests:
   ```bash
   python -m pytest p00_fundamentals/c00_best_practices -v
   ```
3. Run the code-quality tools with the example configuration (the tools read `[tool.*]` sections only from a file
   called `pyproject.toml`, so copy the example first):
   ```bash
   cp p00_fundamentals/c00_best_practices/s00_development_projects/example_pyproject.toml /tmp/pyproject.toml
   ruff check --config /tmp/pyproject.toml p00_fundamentals/c00_best_practices
   ruff format --check --config /tmp/pyproject.toml p00_fundamentals/c00_best_practices
   mypy --config-file /tmp/pyproject.toml p00_fundamentals/c00_best_practices
   ```
4. To use the pre-commit example in your own project, copy `example_pre-commit-config.yaml` to the root of that
   repository as `.pre-commit-config.yaml` and run `pre-commit install` (see the comments in the file).

The theory and a walkthrough are in `book_best_practices.md` and `s00_development_projects/book_software_development.md`;
`s01_acknowledgements/book_acknowledgements.md` covers licences and attribution.

## Files
- `s00_development_projects/main_best_practices.py`: the example script
- `s00_development_projects/utils/scaling.py`: tested feature standardisation and a reproducible train/test split
- `s00_development_projects/utils/config.py`: settings from `GZH_*` environment variables (validated dataclass)
- `s00_development_projects/utils/reproducibility.py`: seeding, timing and run metadata
- `s00_development_projects/test_scaling.py`, `test_config.py`: pytest examples (fixtures, parametrize, tmp_path,
  monkeypatch, caplog, approx, property-style tests, an end-to-end smoke test)
- `s00_development_projects/example_pyproject.toml`: example configuration of ruff, black, pytest and mypy
- `s00_development_projects/example_pre-commit-config.yaml`: example pre-commit hooks (ruff, file hygiene,
  `detect-private-key`)
