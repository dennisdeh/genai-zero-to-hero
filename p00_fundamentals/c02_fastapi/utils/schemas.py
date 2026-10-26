"""
Pydantic schemas (data contracts) of the model-serving API.

The schemas serve three purposes at once:
1. Validation: malformed requests (missing fields, wrong types, values out of range, unknown fields)
   are rejected with an informative HTTP 422 response before they reach the model.
2. Serialisation: responses are converted to JSON in a well-defined shape.
3. Documentation: FastAPI turns the schemas into an OpenAPI specification, visible at /docs.
"""

from pydantic import BaseModel, ConfigDict, Field

# an example flower used in the OpenAPI documentation (and the client script)
EXAMPLE_FLOWER = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


class IrisFeatures(BaseModel):
    """Measurements of a single iris flower in centimetres (one model input)."""

    # extra="forbid": reject unknown fields, e.g. typos such as "sepal_lenght"
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_FLOWER]})

    sepal_length: float = Field(..., gt=0, le=30, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, le=30, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, le=30, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, le=30, description="Petal width in cm")


class BatchPredictionRequest(BaseModel):
    """Several flowers to be classified in a single request."""

    model_config = ConfigDict(extra="forbid")

    instances: list[IrisFeatures] = Field(..., min_length=1, description="Flowers to classify")


class Prediction(BaseModel):
    """The model output for a single flower."""

    label: str = Field(..., description="Predicted species")
    class_index: int = Field(..., ge=0, description="Index of the predicted class")
    probabilities: dict[str, float] = Field(..., description="Predicted probability per species")


class PredictionResponse(BaseModel):
    """Response of the single-prediction endpoint."""

    model_version: str
    prediction: Prediction


class BatchPredictionResponse(BaseModel):
    """Response of the batch-prediction endpoint."""

    model_version: str
    n_instances: int
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    """Liveness information: the process is up and serving HTTP."""

    status: str


class ReadinessResponse(BaseModel):
    """Readiness information: the service is able to answer prediction requests."""

    status: str
    model_loaded: bool
    model_version: str | None = None


class ModelMetadata(BaseModel):
    """Metadata stored next to the model artefact by the training script."""

    # pydantic reserves the "model_" prefix by default; we deliberately use it for "model_name"
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    version: str
    trained_at: str
    framework: str
    sklearn_version: str
    python_version: str
    dataset: str
    features: list[str]
    classes: list[str]
    metrics: dict[str, float]
    n_train: int
    n_test: int
