"""EstateIQ Production Pipeline Artifact Wrapper and Serialization."""

import hashlib
import json
import os
import time
from typing import Dict, Any, Union, Optional
import joblib
import numpy as np
import pandas as pd


class EstateIQInferencePipeline:
    """Unified inference container binding fitted preprocessing, estimator, and target scaling.

    Guarantees:
    - Identical feature transformations applied to inference inputs as in training.
    - Automatic target back-transformation if estimator was trained in log1p space.
    - Non-negative price estimate guarantees.
    - Embedded metadata tracking provenance, evaluation metrics, and dataset fingerprint.
    """

    def __init__(
        self,
        preprocessor: Any,
        estimator: Any,
        use_log1p_target: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.use_log1p_target = use_log1p_target
        self.metadata = metadata or {}

    def predict(self, X: Union[pd.DataFrame, Dict[str, Any], list]) -> np.ndarray:
        """Generate predicted prices in Lakhs INR for input property record(s)."""
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(f"Unsupported input type for prediction: {type(X)}")

        X_proc = self.preprocessor.transform(df)
        raw_pred = self.estimator.predict(X_proc)

        if self.use_log1p_target:
            pred_orig = np.expm1(np.asarray(raw_pred, dtype=float))
        else:
            pred_orig = np.asarray(raw_pred, dtype=float)

        return np.clip(pred_orig, a_min=0.0, a_max=None)


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 fingerprint of a file for metadata provenance."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_production_pipeline_artifact(
    pipeline_artifact: EstateIQInferencePipeline,
    artifact_path: str = "ml/inference/artifacts/estateiq_model_pipeline.joblib",
    metadata_path: str = "ml/inference/artifacts/model_metadata.json",
) -> None:
    """Save the unified inference pipeline artifact and its accompanying metadata JSON."""
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(pipeline_artifact, artifact_path)

    if metadata_path:
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_artifact.metadata, f, indent=2)


def load_production_pipeline_artifact(
    artifact_path: str = "ml/inference/artifacts/estateiq_model_pipeline.joblib",
) -> EstateIQInferencePipeline:
    """Safely load the verified production inference artifact."""
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Model artifact not found at: {artifact_path}. "
            "Inference service requires a trained pipeline artifact."
        )
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, EstateIQInferencePipeline):
        raise TypeError(
            f"Loaded artifact is of unexpected type: {type(artifact)}. "
            "Expected EstateIQInferencePipeline instance."
        )
    return artifact
