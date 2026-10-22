"""Pydantic Schemas for Model Metadata and Evaluation Metrics Endpoint."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ModelMetricsResponse(BaseModel):
    """Schema exposing genuine, verified model evaluation results from successful training runs."""

    model_name: str
    estimator_class: str
    training_timestamp: str
    dataset_sha256: str
    n_train_samples: int
    n_val_samples: int
    val_rmse_lacs: float
    val_mae_lacs: float
    val_r2_score: float
    train_rmse_lacs: float
    train_mae_lacs: float
    train_r2_score: float
    selection_rule: str
    target_currency_unit: str
    status: str = "operational"
    disclaimer: str = (
        "Validation metrics represent empirical performance on held-out dataset records and do not "
        "guarantee future market accuracy or legal appraisal values."
    )
