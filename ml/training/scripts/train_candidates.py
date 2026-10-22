"""Candidate Model Training, Comparison, and Evaluation for EstateIQ.

Trains and evaluates 4 standard regression algorithms across raw and log1p targets:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. K-Nearest Neighbors Regressor

Strict project rules:
- Preprocessing is fitted exclusively on X_train.
- All models are evaluated on the exact same validation partition.
- All evaluation metrics (MAE, RMSE, R²) are computed on the original scale (Lacs).
- No test.csv or sample_submission.csv is ever used for evaluation.
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from ml.common.constants import (
    DEFAULT_TRAIN_CLEANED_PATH,
    DEFAULT_RANDOM_STATE,
    DEFAULT_VAL_SIZE,
    TARGET_COLUMN,
)
from ml.pipeline.data_loader import load_and_split_training_data
from ml.pipeline.preprocessing import build_feature_preprocessor
from ml.pipeline.evaluation import calculate_regression_metrics, inverse_log1p_predictions


def get_candidate_configurations() -> List[Dict[str, Any]]:
    """Define candidate estimators and target scaling modes."""
    return [
        {
            "name": "LinearRegression_Raw",
            "estimator": LinearRegression(),
            "use_log1p_target": False,
        },
        {
            "name": "LinearRegression_Log1p",
            "estimator": LinearRegression(),
            "use_log1p_target": True,
        },
        {
            "name": "DecisionTree_Raw",
            "estimator": DecisionTreeRegressor(
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=DEFAULT_RANDOM_STATE,
            ),
            "use_log1p_target": False,
        },
        {
            "name": "DecisionTree_Log1p",
            "estimator": DecisionTreeRegressor(
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=DEFAULT_RANDOM_STATE,
            ),
            "use_log1p_target": True,
        },
        {
            "name": "RandomForest_Raw",
            "estimator": RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=DEFAULT_RANDOM_STATE,
            ),
            "use_log1p_target": False,
        },
        {
            "name": "RandomForest_Log1p",
            "estimator": RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=DEFAULT_RANDOM_STATE,
            ),
            "use_log1p_target": True,
        },
        {
            "name": "KNN_Raw",
            "estimator": KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
            "use_log1p_target": False,
        },
        {
            "name": "KNN_Log1p",
            "estimator": KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
            "use_log1p_target": True,
        },
    ]


def run_candidate_model_evaluation(
    data_path: str = DEFAULT_TRAIN_CLEANED_PATH,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """Execute reproducible comparison across all candidate regression models."""
    print("=" * 60)
    print("EstateIQ — Candidate Model Training and Evaluation Pipeline")
    print("=" * 60)

    # 1. Load and split data
    print(f"Loading labeled training data from: {data_path}")
    X_train, X_val, y_train, y_val = load_and_split_training_data(
        data_path=data_path,
        val_size=val_size,
        random_state=random_state,
    )

    print(f"Training split:   {X_train.shape[0]} rows")
    print(f"Validation split: {X_val.shape[0]} rows (strictly isolated)")

    # 2. Fit feature preprocessor exclusively on training partition
    print("\nFitting feature preprocessing pipeline on X_train...")
    feature_preprocessor = build_feature_preprocessor(numeric_scaler="robust")
    start_prep = time.time()
    X_train_proc = feature_preprocessor.fit_transform(X_train)
    X_val_proc = feature_preprocessor.transform(X_val)
    prep_time = round(time.time() - start_prep, 3)
    print(f"Preprocessing completed in {prep_time}s. Processed features shape: {X_train_proc.shape}")

    # Prepare targets
    y_train_raw = y_train.values.astype(float)
    y_val_raw = y_val.values.astype(float)

    y_train_log = np.log1p(y_train_raw)
    y_val_log = np.log1p(y_val_raw)

    candidates = get_candidate_configurations()
    results = []

    print("\nEvaluating Candidate Models on the identical validation partition:")
    print("-" * 80)
    print(f"{'Model Name':<24} | {'Val RMSE':<10} | {'Val MAE':<10} | {'Val R²':<8} | {'Train RMSE':<10}")
    print("-" * 80)

    for cfg in candidates:
        name = cfg["name"]
        model = cfg["estimator"]
        use_log = cfg["use_log1p_target"]

        y_tr = y_train_log if use_log else y_train_raw

        # Train model
        t0 = time.time()
        model.fit(X_train_proc, y_tr)
        fit_time = round(time.time() - t0, 3)

        # Predictions on train and validation
        pred_train = model.predict(X_train_proc)
        pred_val = model.predict(X_val_proc)

        # Inverse transformation if model was trained on log1p scale
        if use_log:
            pred_train_orig = inverse_log1p_predictions(pred_train)
            pred_val_orig = inverse_log1p_predictions(pred_val)
        else:
            pred_train_orig = np.clip(pred_train, a_min=0.0, a_max=None)
            pred_val_orig = np.clip(pred_val, a_min=0.0, a_max=None)

        # Compute metrics strictly on original scale (Lacs)
        val_metrics = calculate_regression_metrics(y_val_raw, pred_val_orig)
        train_metrics = calculate_regression_metrics(y_train_raw, pred_train_orig)

        # Overfitting check: ratio of val RMSE to train RMSE
        overfitting_ratio = round(val_metrics["rmse_lacs"] / max(train_metrics["rmse_lacs"], 1e-6), 2)

        model_result = {
            "model_name": name,
            "use_log1p_target": use_log,
            "fit_time_seconds": fit_time,
            "val_rmse_lacs": val_metrics["rmse_lacs"],
            "val_mae_lacs": val_metrics["mae_lacs"],
            "val_r2_score": val_metrics["r2_score"],
            "train_rmse_lacs": train_metrics["rmse_lacs"],
            "train_mae_lacs": train_metrics["mae_lacs"],
            "train_r2_score": train_metrics["r2_score"],
            "overfitting_ratio": overfitting_ratio,
        }
        results.append(model_result)

        print(
            f"{name:<24} | {val_metrics['rmse_lacs']:<10.2f} | "
            f"{val_metrics['mae_lacs']:<10.2f} | {val_metrics['r2_score']:<8.4f} | "
            f"{train_metrics['rmse_lacs']:<10.2f}"
        )

    print("-" * 80)

    # Model Selection Rule: Minimum Validation RMSE
    sorted_by_rmse = sorted(results, key=lambda x: x["val_rmse_lacs"])
    best_candidate = sorted_by_rmse[0]

    print(f"\nWinning Candidate Model: {best_candidate['model_name']}")
    print(f"Selection Metric (Val RMSE): {best_candidate['val_rmse_lacs']} Lacs")
    print(f"Validation MAE:              {best_candidate['val_mae_lacs']} Lacs")
    print(f"Validation R² Score:         {best_candidate['val_r2_score']}")

    # 4. Save Candidate Comparison Experiment Report
    from ml.pipeline.artifact import (
        EstateIQInferencePipeline,
        compute_file_sha256,
        save_production_pipeline_artifact,
    )

    data_fingerprint = compute_file_sha256(data_path) if os.path.exists(data_path) else "unknown"

    summary_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_path": data_path,
        "dataset_sha256": data_fingerprint,
        "n_train_samples": X_train.shape[0],
        "n_val_samples": X_val.shape[0],
        "random_state": random_state,
        "val_size": val_size,
        "selection_rule": "Minimize validation RMSE on original price scale (Lacs)",
        "best_model": best_candidate["model_name"],
        "candidates": results,
    }

    report_dir = "ml/training/experiments"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "candidate_comparison_report.json")
    import json
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nSaved candidate comparison report to: {report_path}")

    # 5. Build and Save Production Candidate Artifact
    # Find winning configuration
    winning_cfg = next(c for c in candidates if c["name"] == best_candidate["model_name"])
    winning_estimator = winning_cfg["estimator"]
    use_log1p = winning_cfg["use_log1p_target"]

    # Package unified pipeline
    artifact_metadata = {
        "model_name": best_candidate["model_name"],
        "estimator_class": winning_estimator.__class__.__name__,
        "estimator_params": str(winning_estimator.get_params()),
        "use_log1p_target": use_log1p,
        "training_timestamp": summary_report["timestamp"],
        "dataset_sha256": data_fingerprint,
        "n_train_samples": X_train.shape[0],
        "n_val_samples": X_val.shape[0],
        "random_state": random_state,
        "val_rmse_lacs": best_candidate["val_rmse_lacs"],
        "val_mae_lacs": best_candidate["val_mae_lacs"],
        "val_r2_score": best_candidate["val_r2_score"],
        "train_rmse_lacs": best_candidate["train_rmse_lacs"],
        "train_mae_lacs": best_candidate["train_mae_lacs"],
        "train_r2_score": best_candidate["train_r2_score"],
        "selection_rule": summary_report["selection_rule"],
        "target_currency_unit": "Lakhs INR (1 Lac = 100,000 INR)",
    }

    production_pipeline = EstateIQInferencePipeline(
        preprocessor=feature_preprocessor,
        estimator=winning_estimator,
        use_log1p_target=use_log1p,
        metadata=artifact_metadata,
    )

    artifact_joblib_path = "ml/inference/artifacts/estateiq_model_pipeline.joblib"
    artifact_meta_path = "ml/inference/artifacts/model_metadata.json"

    save_production_pipeline_artifact(
        pipeline_artifact=production_pipeline,
        artifact_path=artifact_joblib_path,
        metadata_path=artifact_meta_path,
    )
    print(f"Saved production pipeline artifact to: {artifact_joblib_path}")
    print(f"Saved artifact metadata to:           {artifact_meta_path}")

    # Verify reload
    from ml.pipeline.artifact import load_production_pipeline_artifact
    reloaded = load_production_pipeline_artifact(artifact_joblib_path)
    sample_val_pred = reloaded.predict(X_val.iloc[:3])
    print(f"Verified artifact reload test. Sample predictions: {sample_val_pred.round(2).tolist()} Lacs")

    return summary_report


if __name__ == "__main__":
    report = run_candidate_model_evaluation()

