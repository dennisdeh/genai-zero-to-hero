"""
Automated tests of the model-serving API with FastAPI's TestClient (no running server needed).

The TestClient calls the application in-process. Used as a context manager (`with TestClient(app)`),
it also runs the lifespan handler, so the model is loaded exactly as in production. The tests train a
fresh model into a temporary folder, so they do not depend on files created by other scripts.

How to run (from the repository root):
    python -m pytest p00_fundamentals/c02_fastapi -v
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from p00_fundamentals.c02_fastapi.main_api import create_app
from p00_fundamentals.c02_fastapi.utils.schemas import EXAMPLE_FLOWER
from p00_fundamentals.c02_fastapi.utils.training import train_and_save

VERSION = "9.9.9-test"


# 1. Fixtures: a trained model in a temporary folder and clients with/without a model
@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Train a model once for all tests of this module."""
    path = str(tmp_path_factory.mktemp("model"))
    train_and_save(model_dir=path, version=VERSION, verbose=False)
    return path


@pytest.fixture(scope="module")
def client(model_dir: str) -> Iterator[TestClient]:
    """A client of an app with a loaded model."""
    with TestClient(create_app(model_dir=model_dir)) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path) -> Iterator[TestClient]:
    """A client of an app whose model folder is empty."""
    with TestClient(create_app(model_dir=str(tmp_path))) as c:
        yield c


# 2. Operational endpoints
def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert "X-Process-Time-Ms" in resp.headers


def test_ready(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True
    assert resp.json()["model_version"] == VERSION


def test_metadata(client: TestClient) -> None:
    meta = client.get("/v1/model/metadata").json()
    assert meta["version"] == VERSION
    assert meta["classes"] == ["setosa", "versicolor", "virginica"]
    assert meta["metrics"]["test_accuracy"] > 0.85


# 3. Predictions
def test_predict_single(client: TestClient) -> None:
    resp = client.post("/v1/predict", json=EXAMPLE_FLOWER)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == VERSION
    assert body["prediction"]["label"] == "setosa"
    assert sum(body["prediction"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_batch(client: TestClient) -> None:
    virginica = {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.7, "petal_width": 2.3}
    resp = client.post("/v1/predict/batch", json={"instances": [EXAMPLE_FLOWER, virginica]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_instances"] == 2
    assert [p["label"] for p in body["predictions"]] == ["setosa", "virginica"]


def test_batch_size_limit(model_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_BATCH_SIZE", "2")
    with TestClient(create_app(model_dir=model_dir)) as c:
        resp = c.post("/v1/predict/batch", json={"instances": [EXAMPLE_FLOWER] * 3})
    assert resp.status_code == 413


# 4. Input validation
@pytest.mark.parametrize(
    "payload",
    [
        {**EXAMPLE_FLOWER, "sepal_length": -1.0},  # out of range
        {**EXAMPLE_FLOWER, "sepal_length": "long"},  # wrong type
        {k: v for k, v in EXAMPLE_FLOWER.items() if k != "petal_width"},  # missing field
        {**EXAMPLE_FLOWER, "colour": "blue"},  # unknown field
    ],
)
def test_predict_invalid_input(client: TestClient, payload: dict) -> None:
    assert client.post("/v1/predict", json=payload).status_code == 422


def test_empty_batch_rejected(client: TestClient) -> None:
    assert client.post("/v1/predict/batch", json={"instances": []}).status_code == 422


# 5. Behaviour without a model: alive, but not ready
def test_without_model(client_without_model: TestClient) -> None:
    assert client_without_model.get("/health").status_code == 200
    assert client_without_model.get("/ready").status_code == 503
    assert client_without_model.post("/v1/predict", json=EXAMPLE_FLOWER).status_code == 503
