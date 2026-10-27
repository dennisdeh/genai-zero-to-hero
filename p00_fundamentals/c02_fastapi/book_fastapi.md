## Serving models with FastAPI
A trained model is only useful once other programs can use it. The most common way to make a model available
is to wrap it in a web service with an HTTP API: a client (a web front-end, another micro-service, a batch job,
an agent) sends the model inputs as JSON and receives the predictions as JSON. This decouples the model from its
consumers: they do not need Python, scikit-learn or the model file, only the URL and the data contract.

This chapter builds a small but realistic model-serving service with [FastAPI](https://fastapi.tiangolo.com/).
The same service is containerised and deployed to Kubernetes and the cloud in part p01.

### Why FastAPI?
FastAPI is a modern Python web framework built on two libraries:
- **Starlette**: an ASGI (Asynchronous Server Gateway Interface) toolkit handling routing, requests and responses;
- **Pydantic**: data validation through type hints.

Its main advantages for ML engineering are:
- **Type hints as the contract**: the request and response formats are declared once as Pydantic models.
  FastAPI uses them to validate inputs, serialise outputs and generate an OpenAPI specification with
  interactive documentation (Swagger UI at `/docs`) - documentation that is always in sync with the code.
- **Performance**: it runs on an ASGI server such as Uvicorn and supports `async` endpoints, making it one of the
  fastest Python frameworks.
- **Dependency injection**: shared resources (a loaded model, a database connection, settings) are provided to
  endpoints via `Depends`, which keeps endpoints small and testable.
- **Testing**: the `TestClient` calls the application in-process, without starting a server.

Alternatives include Flask (simpler, synchronous, no built-in validation), Django REST framework (full-stack) and
dedicated model servers such as BentoML, Ray Serve, TorchServe, NVIDIA Triton or vLLM (for LLMs).
A hand-written FastAPI service is the most transparent option for learning and is sufficient for many models.

### Anatomy of the service
The chapter consists of the following files:
- `main_train_model.py` with `utils/training.py`: trains the model and saves the *model package*;
- `main_api.py`: the FastAPI application;
- `utils/schemas.py`: the Pydantic schemas (the data contract);
- `utils/model_io.py`: saving/loading the model package and making predictions;
- `utils/config.py`: settings from environment variables;
- `main_client.py`: a client using `requests`;
- `test_api.py`: automated tests with `pytest`.

#### Training and the model package
The model is a scikit-learn `Pipeline` of a `StandardScaler` and a `LogisticRegression`, trained on the iris
dataset (150 flowers, 4 measurements, 3 species), which ships with scikit-learn so no download is needed.
Bundling the preprocessing with the estimator is crucial for serving: the API receives raw measurements and
the identical transformations from training are applied automatically, avoiding *training-serving skew*.

The training script evaluates the model (5-fold cross-validation on the training set plus a hold-out test set)
and saves two files to `outputs/model/`:
- `model.joblib`: the fitted pipeline, serialised with joblib (an efficient pickle for numpy arrays);
- `metadata.json`: model name, semantic version, training time, Python and scikit-learn versions, feature
  names, class names, metrics and data sizes.

The metadata makes the model *traceable*: the API exposes it, so a consumer can always find out which model
version produced a prediction. Two caveats of pickle-based formats are worth remembering:
1. Unpickling executes code, so only load model files from trusted sources.
2. Pickles are not guaranteed to work across library versions. The serving environment must use the same
   scikit-learn version as training; the loader therefore logs a warning on a version mismatch.
   In production, pin versions in the container image, or use a portable format such as ONNX
   or an MLflow model (see p05).

#### Schemas: validation as a first line of defence
The input schema describes one flower:
```python
class IrisFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sepal_length: float = Field(..., gt=0, le=30, description="Sepal length in cm")
    ...
```
Every request body is parsed into this class before the endpoint function runs. Missing fields, wrong types,
values out of range and - because of `extra="forbid"` - unknown fields such as typos are rejected with
HTTP 422 (Unprocessable Content) and a precise error message listing every problem. The model therefore never
sees malformed data. The response schemas (`PredictionResponse`, `BatchPredictionResponse`, `ModelMetadata`)
guarantee that clients always receive the same JSON structure.

#### Loading the model once: the lifespan handler
Loading a model can take from milliseconds (iris) to minutes (an LLM). It must therefore happen once when the
server starts, not on every request. FastAPI's *lifespan* is an async context manager: the code before `yield`
runs at start-up, the code after it at shut-down:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = load_model_bundle(settings.model_dir)
    yield
    app.state.model_bundle = None
```
If the model cannot be found, the service still starts but reports that it is not ready (see below), and the
prediction endpoints answer with HTTP 503 (Service Unavailable) via the `get_model_bundle` dependency.

#### Liveness and readiness
Orchestrators such as Docker, Kubernetes and cloud platforms continuously check the health of a service.
Two different questions are asked:
- **Liveness** (`GET /health`): is the process alive? If not, it should be restarted.
- **Readiness** (`GET /ready`): can the service handle requests right now, i.e. is the model loaded?
  If not, no traffic should be routed to it - but restarting it would not necessarily help.

Separating the two means that a service without a usable model (here: model files missing) is taken out of the
load balancer instead of being restarted over and over, and it is used directly by the Kubernetes probes in p01.
Note that in this implementation the model is loaded in the lifespan, i.e. *before* Uvicorn starts accepting
connections: while a large model is loading, neither `/health` nor `/ready` answers. A liveness probe would then
fail and restart the container in a loop; what prevents this is the *startup probe* in p01, which gives the
application time to start before the liveness probe begins. (Alternatively, the model can be loaded in a
background task, so `/health` answers immediately and `/ready` reports 503 until loading has finished.)
Since the model is only loaded at start-up, a "not ready" instance becomes ready only after a restart, e.g. once
`MODEL_DIR` has been fixed.

#### Versioned endpoints
The prediction endpoints live under the prefix `/v1` (an `APIRouter`):
- `GET /v1/model/metadata`: metadata of the served model;
- `POST /v1/predict`: classify one flower;
- `POST /v1/predict/batch`: classify a list of flowers in one request.

Versioning the API path makes it possible to introduce breaking changes (e.g. a `/v2` with a new input schema)
while existing clients keep using `/v1`. Note that the *API version* (the contract) and the *model version*
(the trained artefact) are different things: a retrained model with the same inputs is a new model version
behind the same API version. Every prediction response contains the model version for traceability.

Batch prediction is more efficient than many single requests because the HTTP overhead is paid once and the
model processes a matrix in a single vectorised call. Its size is limited at two levels: the schema declares
`Field(max_length=MAX_INSTANCES_PER_REQUEST)` (10,000), so larger batches are rejected during validation with
HTTP 422 like any other invalid input, and the endpoint enforces the configurable, lower `MAX_BATCH_SIZE`
(default 1,000) with HTTP 413 (Content Too Large). Both checks run only after the request body has been received
and parsed as JSON, so they cap the validation and inference work, but do not protect against huge request
bodies: a limit on the body size belongs in the server or reverse proxy in front of the API (e.g.
`client_max_body_size` in nginx, or the request size limit of an API gateway or load balancer).

#### Sync or async endpoints?
FastAPI accepts both `def` and `async def` endpoints. An `async def` endpoint runs directly on the event loop,
so any blocking work inside it (such as a CPU-bound `model.predict`) blocks *all* other requests. A plain
`def` endpoint is executed in a thread pool instead. Rule of thumb:
- use `async def` when the endpoint awaits I/O (calling another API, a database with an async driver, an LLM);
- use `def` for CPU-bound model inference, as done here.

For heavier models, scale with several worker processes (`uvicorn --workers N`) or, preferably in containers,
with several replicas behind a load balancer (p01).

#### Middleware and configuration
A middleware wraps every request; here it measures the processing time and returns it in the
`X-Process-Time-Ms` header - a simple form of observability. In production one would add structured logging,
request IDs and metrics (e.g. Prometheus).

All settings (`MODEL_DIR`, `LOG_LEVEL`, `MAX_BATCH_SIZE`) are read from environment variables, following the
*twelve-factor app* principle of storing configuration in the environment. The same code then runs unchanged
locally, in a container and in Kubernetes, where the variables are set by a ConfigMap. The values are validated
when the settings are created: an unknown `LOG_LEVEL` falls back to `INFO` with a warning (otherwise
`logging.basicConfig` would raise an error at import time and the container would end up in a crash loop),
while an invalid `MAX_BATCH_SIZE` stops the start-up with a clear error message.

The application is created by a factory function `create_app(model_dir=None)`. The module-level
`app = create_app()` is what Uvicorn imports, while tests create their own instances pointing at a temporary
model folder.

### Running the service
From the repository root:
```bash
python -m p00_fundamentals.c02_fastapi.main_train_model
uvicorn p00_fundamentals.c02_fastapi.main_api:app --port 5000
python -m p00_fundamentals.c02_fastapi.main_client       # in a second terminal
```
Uvicorn is the ASGI server: it listens on the port, parses HTTP and hands requests to the application.
By default, it binds to `127.0.0.1`, i.e. it is only reachable from the same machine; inside a container it must
bind to `0.0.0.0` (see p01). The client prints every response; a single prediction looks like:
```json
{
  "model_version": "1.0.0",
  "prediction": {
    "label": "setosa",
    "class_index": 0,
    "probabilities": {"setosa": 0.98, "versicolor": 0.02, "virginica": 0.0}
  }
}
```

### Testing the API
`test_api.py` uses FastAPI's `TestClient`, which is based on `httpx` and calls the application in-process.
Used as a context manager (`with TestClient(app) as client`), it also runs the lifespan, so the model is loaded
exactly as in production. A module-scoped fixture trains a fresh model into a temporary folder, which makes the
tests independent of any previously generated files. The tests cover:
- the operational endpoints (liveness, readiness, metadata);
- single and batch predictions, including sanity checks on the predicted labels and probabilities;
- input validation (out-of-range values, wrong types, missing and unknown fields, empty batches);
- the batch size limit (configured via `monkeypatch.setenv`);
- the behaviour without a model (alive, not ready, HTTP 503).

Run them with `python -m pytest p00_fundamentals/c02_fastapi -v`. Such tests are the first stage of the CI
pipeline sketched in p01: no image is built or deployed unless they pass.

### From prototype to production
The service is deliberately small, but its structure carries over to larger models. Topics for a production
deployment, several of them covered in later parts, include:
- authentication (API keys, OAuth2) and rate limiting;
- model registries and automated retraining (MLflow, p05);
- monitoring of latency, errors and data/prediction drift;
- containerisation, orchestration and scaling (p01).
