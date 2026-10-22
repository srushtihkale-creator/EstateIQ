"""Integration tests for Health, Readiness, and Model Metadata API routes."""

import pytest
from fastapi import status


def test_health_check_endpoint(client):
    """Verify /health returns 200 OK and reports active database connectivity."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"
    assert "version" in data


def test_readiness_check_endpoint(client):
    """Verify /ready probe confirms both database connectivity and model artifact readiness."""
    response = client.get("/ready")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ready"
    assert data["database"] == "connected"
    assert data["model_artifact"] == "loaded"
    assert data["model_name"] == "DecisionTree_Raw"


def test_model_metrics_endpoint(client):
    """Verify /api/model/metrics returns verified evaluation metadata without training."""
    response = client.get("/api/model/metrics")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["model_name"] == "DecisionTree_Raw"
    assert data["estimator_class"] == "DecisionTreeRegressor"
    assert data["val_rmse_lacs"] > 0
    assert data["val_r2_score"] > 0.80
    assert data["n_train_samples"] == 23240
    assert data["n_val_samples"] == 5810
    assert "disclaimer" in data
