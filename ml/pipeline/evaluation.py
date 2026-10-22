"""Evaluation utilities, regression metrics, and reporting for EstateIQ."""

import json
from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> Dict[str, float]:
    """Calculate genuine regression metrics in the original target scale (Lacs).

    Metrics:
    - MAE: Mean Absolute Error (in Lacs)
    - RMSE: Root Mean Squared Error (in Lacs)
    - R2: Coefficient of Determination (R²)

    Note: As mandated by AGENTS.md, PRD, and SRS, regression metrics must NEVER
    be referred to as an 'accuracy percentage'.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    # Hard guard against NaN or infinite values
    if np.isnan(y_p).any() or np.isinf(y_p).any():
        raise ValueError("Model predictions contain NaN or Infinite values.")

    # Prevent negative price predictions on original scale
    y_p_clipped = np.clip(y_p, a_min=0.0, a_max=None)

    mae = float(mean_absolute_error(y_t, y_p_clipped))
    # root_mean_squared_error is standard in scikit-learn >= 1.4
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p_clipped)))
    r2 = float(r2_score(y_t, y_p_clipped))

    return {
        "mae_lacs": round(mae, 4),
        "rmse_lacs": round(rmse, 4),
        "r2_score": round(r2, 4),
    }


def inverse_log1p_predictions(y_pred_log: np.ndarray) -> np.ndarray:
    """Safely back-transform log1p predictions to the original price scale (Lacs).

    Transformation: y_raw = exp(y_log) - 1, clipped to >= 0.
    """
    y_raw = np.expm1(np.asarray(y_pred_log, dtype=float))
    return np.clip(y_raw, a_min=0.0, a_max=None)


def save_metrics_report(
    report_data: Dict[str, Any],
    output_path: str,
) -> None:
    """Save evaluation metrics and model comparison metadata to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
