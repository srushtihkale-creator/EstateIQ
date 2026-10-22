"""Model Metadata and Evaluation Metrics API Router."""

import json
import os
from fastapi import APIRouter, HTTPException, status
from backend.app.core.config import settings
from backend.app.schemas.model import ModelMetricsResponse

router = APIRouter(prefix="/model", tags=["Model Information"])


@router.get(
    "/metrics",
    response_model=ModelMetricsResponse,
    summary="Get verified model training and evaluation metrics",
)
def get_model_metrics():
    """Return genuine validation metrics produced during the verified training run.

    Does not trigger training or generate synthetic metrics.
    """
    meta_path = settings.MODEL_METADATA_PATH
    if not os.path.exists(meta_path):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model evaluation metadata report is not available.",
        )

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return ModelMetricsResponse(
            model_name=data.get("model_name", "DecisionTree_Raw"),
            estimator_class=data.get("estimator_class", "DecisionTreeRegressor"),
            training_timestamp=data.get("training_timestamp", "unknown"),
            dataset_sha256=data.get("dataset_sha256", "unknown"),
            n_train_samples=data.get("n_train_samples", 0),
            n_val_samples=data.get("n_val_samples", 0),
            val_rmse_lacs=data.get("val_rmse_lacs", 0.0),
            val_mae_lacs=data.get("val_mae_lacs", 0.0),
            val_r2_score=data.get("val_r2_score", 0.0),
            train_rmse_lacs=data.get("train_rmse_lacs", 0.0),
            train_mae_lacs=data.get("train_mae_lacs", 0.0),
            train_r2_score=data.get("train_r2_score", 0.0),
            selection_rule=data.get("selection_rule", "Minimize validation RMSE"),
            target_currency_unit=data.get("target_currency_unit", "Lakhs INR"),
            status="operational",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading model metadata: {str(e)}",
        )
