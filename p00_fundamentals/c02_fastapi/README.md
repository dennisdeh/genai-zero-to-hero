## Environment
- Python conda environment to use: `conda_env_gzh_api0.yml`
  (create it with `conda env create -f p00_fundamentals/c02_fastapi/conda_env_gzh_api0.yml`)
- Docker compose setup: none in this chapter; the containerised version of this service is built in
  `p01_containerisation_and_cloud_deployment/c00_docker`.
- Environment variables (all optional):
  - `MODEL_DIR`: folder with `model.joblib` and `metadata.json` (default: `p00_fundamentals/c02_fastapi/outputs/model`)
  - `LOG_LEVEL`: logging level of the API (default: `INFO`)
  - `MAX_BATCH_SIZE`: maximum number of instances per batch request (default: `1000`)
  - `HOST`, `PORT`: bind address when started with `python -m ...main_api` (default: `127.0.0.1`, `5000`)
  - `API_URL`: base URL used by the client (default: `http://localhost:5000`)

## How to run
All commands are run from the repository root.

1. Train the model (takes a few seconds; writes to `p00_fundamentals/c02_fastapi/outputs/model`):
   ```bash
   python -m p00_fundamentals.c02_fastapi.main_train_model
   ```
2. Start the API:
   ```bash
   uvicorn p00_fundamentals.c02_fastapi.main_api:app --port 5000
   # during development, auto-reload on code changes:
   uvicorn p00_fundamentals.c02_fastapi.main_api:app --port 5000 --reload
   ```
   Interactive documentation: http://localhost:5000/docs (Swagger UI) and http://localhost:5000/redoc
3. In a second terminal, call the API with the client:
   ```bash
   python -m p00_fundamentals.c02_fastapi.main_client
   ```
   or with curl:
   ```bash
   curl -X POST http://localhost:5000/v1/predict -H "Content-Type: application/json" \
        -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
   ```
4. Run the automated tests (they train their own model in a temporary folder):
   ```bash
   python -m pytest p00_fundamentals/c02_fastapi -v
   ```

## Files
- `main_train_model.py`: trains and saves the scikit-learn pipeline and its metadata
- `main_api.py`: the FastAPI application (lifespan model loading, `/health`, `/ready`, `/v1/...`)
- `main_client.py`: a `requests`-based client calling all endpoints
- `test_api.py`: pytest tests using FastAPI's `TestClient`
- `utils/`: configuration, pydantic schemas, model saving/loading and training code
