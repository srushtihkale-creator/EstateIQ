"""Integration tests for Property Prediction API routes and validations."""

import pytest
from fastapi import status
from backend.app.core.config import settings


@pytest.fixture
def valid_property_payload():
    """Standard valid property prediction payload."""
    return {
        "POSTED_BY": "Owner",
        "UNDER_CONSTRUCTION": 0,
        "RERA": 1,
        "BHK_NO.": 2,
        "BHK_OR_RK": "BHK",
        "SQUARE_FT": 1250.0,
        "READY_TO_MOVE": 1,
        "RESALE": 1,
        "ADDRESS": "Whitefield, Bangalore",
        "LONGITUDE": 77.59796,
        "LATITUDE": 12.96991,
    }


def test_predict_price_as_guest(client, valid_property_payload):
    """Verify guests can obtain price estimates without saving to database history."""
    response = client.post("/api/predictions", json=valid_property_payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["estimated_price_lacs"] > 0
    assert data["formatted_price"].startswith("₹")
    assert data["price_unit"] == "Lakhs INR"
    assert data["model_version"] == "DecisionTree_Raw"
    assert "educational and decision-support" in data["disclaimer"]
    # Guest prediction is not saved to history
    assert data["prediction_id"] is None


def test_predict_price_as_authenticated_user(client, authenticated_user_a, valid_property_payload):
    """Verify authenticated user predictions are automatically persisted to their private history."""
    client.cookies.set(settings.SESSION_COOKIE_NAME, authenticated_user_a["session_id"])

    response = client.post("/api/predictions", json=valid_property_payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["estimated_price_lacs"] > 0
    # Must contain integer prediction_id referencing saved record
    assert isinstance(data["prediction_id"], int)
    prediction_id = data["prediction_id"]

    # Verify record appears in user's history
    history_res = client.get("/api/predictions")
    assert history_res.status_code == status.HTTP_200_OK
    history_data = history_res.json()
    assert history_data["total"] >= 1
    assert any(item["id"] == prediction_id for item in history_data["items"])


def test_predict_price_missing_coordinates(client, valid_property_payload):
    """Verify prediction works smoothly when coordinates are missing (imputed by pipeline)."""
    valid_property_payload["LONGITUDE"] = None
    valid_property_payload["LATITUDE"] = None

    response = client.post("/api/predictions", json=valid_property_payload)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["estimated_price_lacs"] > 0


def test_predict_price_unseen_city(client, valid_property_payload):
    """Verify prediction handles completely novel/unseen cities by mapping to 'Other'."""
    valid_property_payload["ADDRESS"] = "Main Market Road, NovelCity999"

    response = client.post("/api/predictions", json=valid_property_payload)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["estimated_price_lacs"] > 0


def test_predict_price_validation_errors(client):
    """Verify Pydantic rejects invalid inputs with HTTP 422 and structured details."""
    # Negative square feet
    payload_bad_area = {
        "POSTED_BY": "Owner",
        "UNDER_CONSTRUCTION": 0,
        "RERA": 1,
        "BHK_NO.": 2,
        "BHK_OR_RK": "BHK",
        "SQUARE_FT": -500.0,
        "RESALE": 1,
        "ADDRESS": "Whitefield, Bangalore",
    }
    res1 = client.post("/api/predictions", json=payload_bad_area)
    assert res1.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Invalid BHK (> 20)
    payload_bad_bhk = {
        "POSTED_BY": "Owner",
        "UNDER_CONSTRUCTION": 0,
        "RERA": 1,
        "BHK_NO.": 25,
        "BHK_OR_RK": "BHK",
        "SQUARE_FT": 1000.0,
        "RESALE": 1,
        "ADDRESS": "Whitefield, Bangalore",
    }
    res2 = client.post("/api/predictions", json=payload_bad_bhk)
    assert res2.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Blank address
    payload_bad_addr = {
        "POSTED_BY": "Owner",
        "UNDER_CONSTRUCTION": 0,
        "RERA": 1,
        "BHK_NO.": 2,
        "BHK_OR_RK": "BHK",
        "SQUARE_FT": 1000.0,
        "RESALE": 1,
        "ADDRESS": "   ",
    }
    res3 = client.post("/api/predictions", json=payload_bad_addr)
    assert res3.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
