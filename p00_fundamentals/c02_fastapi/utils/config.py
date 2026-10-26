"""
Runtime configuration of the model-serving API.

All settings are read from environment variables (with sensible defaults), following the
"twelve-factor app" principle: the same code/container image runs unchanged on a laptop, in
docker compose, in Kubernetes (via a ConfigMap) or on a managed cloud service, and only the
environment differs.

Environment variables:
- MODEL_DIR:       folder containing model.joblib and metadata.json (default: <chapter>/outputs/model)
- LOG_LEVEL:       python logging level, e.g. DEBUG, INFO, WARNING (default: INFO)
- MAX_BATCH_SIZE:  maximum number of instances accepted by the batch endpoint (default: 1000)
"""

import os
from dataclasses import dataclass

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
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "1000")),
        )
