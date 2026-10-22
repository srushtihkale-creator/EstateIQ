"""Unit tests for ML evaluation metrics, log1p transformation, and pipeline artifact handling."""

import os
import numpy as np
import pytest
import joblib
from sklearn.linear_model import LinearRegression

from ml.pipeline.evaluation import (
    calculate_regression_metrics,
    inverse_log1p_predictions,
    save_metrics_report,
)
from ml.training.scripts.train_candidates import get_candidate_configurations


def test_calculate_regression_metrics_known_values():
    """Verify calculation of MAE, RMSE, and R2 against known analytical values."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 37.0])

    # Absolute errors: [2, 2, 3, 3] -> MAE = 2.5
    # Squared errors: [4, 4, 9, 9] -> MSE = 6.5 -> RMSE = sqrt(6.5) ≈ 2.5495
    metrics = calculate_regression_metrics(y_true, y_pred)

    assert metrics["mae_lacs"] == 2.5
    assert abs(metrics["rmse_lacs"] - 2.5495) < 0.001
    assert metrics["r2_score"] > 0.90


def test_calculate_regression_metrics_rejects_nan_and_inf():
    """Verify metrics calculation strictly rejects NaN and infinite values."""
    y_true = np.array([10.0, 20.0])
    y_nan = np.array([10.0, np.nan])
    y_inf = np.array([10.0, np.inf])

    with pytest.raises(ValueError, match="NaN or Infinite"):
        calculate_regression_metrics(y_true, y_nan)

    with pytest.raises(ValueError, match="NaN or Infinite"):
        calculate_regression_metrics(y_true, y_inf)


def test_inverse_log1p_predictions():
    """Verify inverse log1p transformation correctly exponentiates and clips negative values."""
    # log1p(0) = 0 -> expm1(0) = 0
    # log1p(99) = ln(100) ≈ 4.60517 -> expm1(4.60517) ≈ 99.0
    y_log = np.array([0.0, np.log1p(99.0), -1.0])
    y_orig = inverse_log1p_predictions(y_log)

    assert abs(y_orig[0] - 0.0) < 1e-5
    assert abs(y_orig[1] - 99.0) < 1e-4
    # Negative exponentiation must be clipped to 0.0
    assert y_orig[2] == 0.0


def test_candidate_configurations_completeness():
    """Verify that all 4 required candidate algorithms are configured."""
    candidates = get_candidate_configurations()
    names = [c["name"] for c in candidates]

    # Check for presence of all four required algorithms
    assert any("LinearRegression" in n for n in names)
    assert any("DecisionTree" in n for n in names)
    assert any("RandomForest" in n for n in names)
    assert any("KNN" in n for n in names)

    # Check both raw and log1p modes
    assert any(c["use_log1p_target"] is True for c in candidates)
    assert any(c["use_log1p_target"] is False for c in candidates)


def test_artifact_serialization_and_deserialization(tmp_path):
    """Verify a fitted model can be saved via joblib and reloaded with identical predictions."""
    X_synthetic = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_synthetic = np.array([10.0, 20.0, 30.0])

    model = LinearRegression()
    model.fit(X_synthetic, y_synthetic)

    original_preds = model.predict(X_synthetic)

    artifact_file = tmp_path / "test_model_pipeline.joblib"
    joblib.dump(model, artifact_file)

    assert os.path.exists(artifact_file)

    reloaded_model = joblib.load(artifact_file)
    reloaded_preds = reloaded_model.predict(X_synthetic)

    np.testing.assert_allclose(original_preds, reloaded_preds, rtol=1e-5)


def test_production_artifact_end_to_end():
    """Verify the saved production pipeline artifact loads and predicts on sample input."""
    from ml.pipeline.artifact import load_production_pipeline_artifact

    artifact_path = "ml/inference/artifacts/estateiq_model_pipeline.joblib"
    assert os.path.exists(artifact_path), "Production artifact must exist after Phase 4B."

    pipeline = load_production_pipeline_artifact(artifact_path)
    assert pipeline.metadata["model_name"] == "DecisionTree_Raw"
    assert pipeline.metadata["val_rmse_lacs"] > 0

    sample_input = {
        "POSTED_BY": "Owner",
        "UNDER_CONSTRUCTION": 0,
        "RERA": 1,
        "BHK_NO.": 2,
        "BHK_OR_RK": "BHK",
        "SQUARE_FT": 1200.0,
        "RESALE": 1,
        "ADDRESS": "Whitefield, Bangalore",
        "LONGITUDE": 77.59796,
        "LATITUDE": 12.96991,
    }

    preds = pipeline.predict(sample_input)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 1
    assert preds[0] > 0.0

