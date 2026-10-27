"""
A small but production-style model-serving API built with FastAPI.

The API serves the iris classifier trained by main_train_model.py and demonstrates:
- loading the model once at start-up with a lifespan handler (instead of on every request)
- pydantic request/response schemas: automatic validation, serialisation and OpenAPI docs (/docs)
- separate liveness (/health) and readiness (/ready) endpoints, as used by Docker and Kubernetes
- a versioned API (/v1/...) with single and batch prediction and a model metadata endpoint
- dependency injection (Depends) to access the loaded model and fail cleanly (HTTP 503) without it
- a small middleware that measures the processing time of each request
- configuration through environment variables (MODEL_DIR, LOG_LEVEL, MAX_BATCH_SIZE)

How to run (from the repository root, after training the model with main_train_model.py):
    uvicorn p00_fundamentals.c02_fastapi.main_api:app --port 5000
    # or: python -m p00_fundamentals.c02_fastapi.main_api
Then open http://localhost:5000/docs for the interactive documentation, or run main_client.py.
"""

import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, status

from p00_fundamentals.c02_fastapi.utils.config import Settings
from p00_fundamentals.c02_fastapi.utils.model_io import ModelBundle, load_model_bundle
from p00_fundamentals.c02_fastapi.utils.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    IrisFeatures,
    ModelMetadata,
    PredictionResponse,
    ReadinessResponse,
)

API_VERSION = "1.0.0"
logger = logging.getLogger("iris_api")


# 1. Dependencies: give the endpoints access to the loaded model and the settings
def get_settings(request: Request) -> Settings:
    """Return the settings stored on the application state."""
    return request.app.state.settings


def get_model_bundle(request: Request) -> ModelBundle:
    """Return the loaded model, or answer with HTTP 503 (Service Unavailable) if there is none."""
    bundle: ModelBundle | None = request.app.state.model_bundle
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Train it with main_train_model.py and check MODEL_DIR.",
        )
    return bundle


ModelDep = Annotated[ModelBundle, Depends(get_model_bundle)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


# 2. Endpoints of the versioned API (all prefixed with /v1)
router_v1 = APIRouter(prefix="/v1", tags=["v1"])


@router_v1.get("/model/metadata", response_model=ModelMetadata)
def model_metadata(bundle: ModelDep) -> ModelMetadata:
    """Return the metadata of the served model (version, training date, metrics, ...)."""
    return bundle.metadata


# note: the prediction endpoints are plain (non-async) functions on purpose. FastAPI runs them in a
# thread pool, so the CPU-bound model call does not block the event loop for other requests.
@router_v1.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures, bundle: ModelDep) -> PredictionResponse:
    """Classify a single flower."""
    prediction = bundle.predict([features])[0]
    return PredictionResponse(model_version=bundle.metadata.version, prediction=prediction)


@router_v1.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    batch: BatchPredictionRequest, bundle: ModelDep, settings: SettingsDep
) -> BatchPredictionResponse:
    """Classify several flowers in one request (more efficient than many single requests)."""
    # the schema already rejects more than MAX_INSTANCES_PER_REQUEST flowers (HTTP 422) during validation;
    # the configurable, lower MAX_BATCH_SIZE is checked here (HTTP 413). Neither limits the size of the
    # request body itself: that belongs in the server or reverse proxy (e.g. client_max_body_size in nginx).
    n = len(batch.instances)
    if n > settings.max_batch_size:
        raise HTTPException(
            status_code=413,  # Content Too Large
            detail=f"Batch of {n} instances exceeds MAX_BATCH_SIZE={settings.max_batch_size}",
        )
    predictions = bundle.predict(batch.instances)
    return BatchPredictionResponse(
        model_version=bundle.metadata.version, n_instances=n, predictions=predictions
    )


# 3. Application factory: lifespan (start-up/shut-down), middleware, operational endpoints
def create_app(model_dir: str | None = None) -> FastAPI:
    """
    Create the FastAPI application.

    :param model_dir: folder with the model artefacts; overrides the MODEL_DIR environment variable
                      (useful in tests)
    """
    settings = Settings.from_env()
    if model_dir is not None:
        settings = Settings(
            model_dir=model_dir,
            log_level=settings.log_level,
            max_batch_size=settings.max_batch_size,
        )
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Code before `yield` runs once at start-up, code after it once at shut-down."""
        app.state.settings = settings
        try:
            app.state.model_bundle = load_model_bundle(settings.model_dir)
            meta = app.state.model_bundle.metadata
            logger.info("Loaded model %s v%s from %s", meta.model_name, meta.version, settings.model_dir)
        except FileNotFoundError as e:
            # keep the process alive (liveness OK), but report "not ready": the model is only loaded here at
            # start-up, so the pod/container must be restarted (or replaced) once MODEL_DIR is fixed
            app.state.model_bundle = None
            logger.error("Could not load model: %s", e)
        yield
        logger.info("Shutting down, releasing the model")
        app.state.model_bundle = None

    app = FastAPI(
        title="Iris classifier API",
        description="Serves a scikit-learn pipeline trained on the iris dataset.",
        version=API_VERSION,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def add_process_time_header(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Measure how long each request takes and return it in the X-Process-Time-Ms header."""
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
        return response

    @app.get("/", tags=["info"])
    def root() -> dict[str, str]:
        """Short description of the service with links to the documentation."""
        return {"service": app.title, "api_version": API_VERSION, "docs": "/docs"}

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        """Liveness probe: the process is running and able to answer HTTP requests."""
        return HealthResponse(status="ok")

    @app.get(
        "/ready",
        response_model=ReadinessResponse,
        tags=["operations"],
        responses={503: {"model": ReadinessResponse}},
    )
    def ready(request: Request, response: Response) -> ReadinessResponse:
        """Readiness probe: the model is loaded and predictions can be served."""
        bundle: ModelBundle | None = request.app.state.model_bundle
        if bundle is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadinessResponse(status="not ready", model_loaded=False)
        return ReadinessResponse(
            status="ready", model_loaded=True, model_version=bundle.metadata.version
        )

    app.include_router(router_v1)
    return app


# the module-level application object used by `uvicorn p00_fundamentals.c02_fastapi.main_api:app`
app = create_app()


if __name__ == "__main__":
    # 4. Run the server directly with python (host/port configurable, e.g. HOST=0.0.0.0 in a container)
    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        log_level=Settings.from_env().log_level.lower(),  # validated, see utils/config.py
    )
