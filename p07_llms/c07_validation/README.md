## Environment
- Python conda environment to use: `conda_env_gzh_llm7.yml`
- Docker compose setup: `p07_llms/c01_running_llms/s01_ollama/docker-compose.yml` (local Ollama on
  `http://localhost:07011`), with a model pulled, e.g. `qwen3:8b`.
- Environment variables (all optional, no `.env` file needed):
  - `OLLAMA_BASE_URL` (default `http://localhost:07011`), `OLLAMA_MODEL` (default `qwen3:8b`)
  - `OLLAMA_JUDGE_MODEL`: a different (ideally stronger) model for the LLM judge; default = `OLLAMA_MODEL`
  - `LLM_TO_USE=openai` and `OPENAI_API_KEY` to use OpenAI instead of Ollama
- MLflow stores runs in `p07_llms/c07_validation/outputs/mlflow.db` (SQLite) and artefacts in
  `outputs/mlartifacts/`.

## How to run
Run from the repository root:

```bash
# evaluation runs logged to MLflow
python -m p07_llms.c07_validation.main_1_evaluate_with_mlflow
python -m p07_llms.c07_validation.main_2_mlflow_genai_evaluate

# look at the results
mlflow ui --backend-store-uri sqlite:///p07_llms/c07_validation/outputs/mlflow.db

# tests: metrics (offline) and LLM regression tests (skipped if no LLM is reachable)
python -m pytest p07_llms/c07_validation/tests -v
```

| File | Shows |
|---|---|
| `main_1_evaluate_with_mlflow.py` | two prompt variants scored with exact match, contains-answer, token F1, ROUGE-L and an LLM judge; params, metrics, prompt and per-question table logged to MLflow |
| `main_2_mlflow_genai_evaluate.py` | the same with `mlflow.genai.evaluate`, custom `@scorer` functions and traces |
| `tests/test_metrics.py` | unit tests of the metric implementations (offline) |
| `tests/test_llm_regression.py` | regression tests: answer format, structured output schema, prompt injection, data leakage, quality gate |
| `utils/` | evaluation set, metrics, rubric judge (`with_structured_output`), MLflow set-up |
