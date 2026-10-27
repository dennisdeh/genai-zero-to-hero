"""
Runtime configuration of the model-serving API.

All settings are read from environment variables (with sensible defaults), following the
"twelve-factor app" principle: the same code/container image runs unchanged on a laptop, in
docker compose, in Kubernetes (via a ConfigMap) or on a managed cloud service, and only the
environment differs.

Environment variables:
- MODEL_DIR:       folder containing model.joblib and metadata.json (default: <chapter>/outputs/model)
- LOG_LEVEL:       python logging level, one of LOG_LEVELS (default: INFO); an unknown value falls back
                   to INFO with a warning, so a typo cannot crash the service at start-up
- MAX_BATCH_SIZE:  maximum number of instances accepted by the batch endpoint (default: 1000); must be an
                   integer between 1 and MAX_INSTANCES_PER_REQUEST (schemas.py), otherwise start-up fails
                   with a clear error message
"""

import logging
import os
from dataclasses import dataclass

from p00_fundamentals.c02_fastapi.utils.schemas import MAX_INSTANCES_PER_REQUEST

logger = logging.getLogger("iris_api")

# logging levels accepted for LOG_LEVEL (also understood by uvicorn's --log-level in lower case)
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# default location of the trained model: an outputs/ folder next to the chapter code (gitignored)
CHAPTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(CHAPTER_DIR, "outputs", "model")

# file names of the model artefacts inside MODEL_DIR
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class Settings:
    """Immutable container for the API settings."""

    model_dir: str = DEFAULT_MODEL_DIR
    log_level: str = "INFO"
    max_batch_size: int = 1000

    @classmethod
    def from_env(cls) -> "Settings":
        """Build the settings from environment variables, falling back to the defaults."""
        return cls(
            model_dir=os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR),
            log_level=parse_log_level(os.getenv("LOG_LEVEL", "INFO")),
            max_batch_size=parse_max_batch_size(os.getenv("MAX_BATCH_SIZE", "1000")),
        )


def parse_log_level(value: str) -> str:
    """
    Return the upper-case logging level name, or INFO (with a warning) if the value is not a level name.

    Without this check, an invalid value such as LOG_LEVEL=verbose makes logging.basicConfig raise a
    ValueError when the module is imported, and the container would end up in a crash loop.
    """
    level = value.strip().upper()
    if level not in LOG_LEVELS:
        logger.warning("Invalid LOG_LEVEL=%r, using INFO instead", value)
        return "INFO"
    return level


def parse_max_batch_size(value: str) -> int:
    """Return MAX_BATCH_SIZE as an integer, failing fast with a clear message if it is invalid."""
    try:
        size = int(value)
    except ValueError:
        raise ValueError(f"MAX_BATCH_SIZE must be an integer, got {value!r}") from None
    if not 1 <= size <= MAX_INSTANCES_PER_REQUEST:
        raise ValueError(
            f"MAX_BATCH_SIZE must be between 1 and {MAX_INSTANCES_PER_REQUEST}, got {size}"
        )
    return size
