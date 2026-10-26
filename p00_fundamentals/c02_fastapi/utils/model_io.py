"""
Saving and loading of the model artefact (a scikit-learn pipeline) together with its metadata.

The model is persisted with joblib (an efficient pickle for objects containing numpy arrays) and the
metadata as human-readable JSON. Keeping both in one folder makes the "model package" easy to copy
into a container image, mount as a volume or upload to object storage.

Security note: unpickling executes code, so only ever load joblib/pickle files from trusted sources.
"""

import json
import logging
import os
from dataclasses import dataclass

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from p00_fundamentals.c02_fastapi.utils.config import METADATA_FILENAME, MODEL_FILENAME
from p00_fundamentals.c02_fastapi.utils.schemas import IrisFeatures, ModelMetadata, Prediction

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """A loaded model together with its metadata."""

    pipeline: Pipeline
    metadata: ModelMetadata

    def predict(self, instances: list[IrisFeatures]) -> list[Prediction]:
        """
        Classify a list of flowers.

        The inputs are converted to a DataFrame whose column names match the names seen during
        training, so the column order of the request does not matter.
        """
        x = pd.DataFrame([inst.model_dump() for inst in instances], columns=self.metadata.features)
        probas = self.pipeline.predict_proba(x)
        predictions = []
        for row in probas:
            idx = int(row.argmax())
            predictions.append(
                Prediction(
                    label=self.metadata.classes[idx],
                    class_index=idx,
                    probabilities={
                        c: round(float(p), 6) for c, p in zip(self.metadata.classes, row)
                    },
                )
            )
        return predictions


def save_model_bundle(pipeline: Pipeline, metadata: ModelMetadata, model_dir: str) -> str:
    """Save the pipeline (joblib) and its metadata (JSON) into model_dir; return the folder path."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(model_dir, MODEL_FILENAME))
    with open(os.path.join(model_dir, METADATA_FILENAME), "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2)
    return model_dir


def load_model_bundle(model_dir: str) -> ModelBundle:
    """
    Load the pipeline and its metadata from model_dir.

    Raises FileNotFoundError if the artefacts do not exist. Logs a warning if the model was trained
    with a different scikit-learn version than the one installed (pickles are not guaranteed to be
    compatible across versions - pin the version in the serving environment!).
    """
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    metadata_path = os.path.join(model_dir, METADATA_FILENAME)
    for path in (model_path, metadata_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model artefact not found: {path}")

    with open(metadata_path, encoding="utf-8") as f:
        metadata = ModelMetadata(**json.load(f))
    pipeline = joblib.load(model_path)

    if metadata.sklearn_version != sklearn.__version__:
        logger.warning(
            "Model trained with scikit-learn %s but %s is installed",
            metadata.sklearn_version,
            sklearn.__version__,
        )
    return ModelBundle(pipeline=pipeline, metadata=metadata)
