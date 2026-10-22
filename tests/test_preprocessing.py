"""Unit tests for EstateIQ data preprocessing, schema validation, and leakage prevention."""

import numpy as np
import pytest
from pydantic import ValidationError

from ml.common.constants import (
    TARGET_COLUMN,
    COLLINEAR_DROPPED_COLUMN,
    INDIA_BOUNDING_BOX,
)
from ml.pipeline.schemas import PropertyInputSchema, PredictionResponseSchema
from ml.pipeline.preprocessing import (
    extract_city_from_address,
    is_coordinate_in_india,
    TargetLeakageGuard,
    CollinearFeatureDropper,
    CoordinateValidator,
    CityExtractor,
)
from ml.pipeline.data_loader import (
    train_val_split_indices,
    separate_features_and_target_records,
)


# ==================== In-Memory Test Fixtures ====================

@pytest.fixture
def sample_valid_property_dict():
    """Minimal in-memory fixture for a valid property input."""
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


@pytest.fixture
def sample_property_records():
    """Minimal in-memory records mimicking raw dataset rows."""
    return [
        {
            "POSTED_BY": "Owner",
            "UNDER_CONSTRUCTION": "0",
            "RERA": "1",
            "BHK_NO.": "2",
            "BHK_OR_RK": "BHK",
            "SQUARE_FT": "1200.0",
            "READY_TO_MOVE": "1",
            "RESALE": "1",
            "ADDRESS": "Whitefield, Bangalore",
            "LONGITUDE": "77.59",
            "LATITUDE": "12.97",
            TARGET_COLUMN: "65.0",
        },
        {
            "POSTED_BY": "Dealer",
            "UNDER_CONSTRUCTION": "1",
            "RERA": "0",
            "BHK_NO.": "3",
            "BHK_OR_RK": "BHK",
            "SQUARE_FT": "1650.0",
            "READY_TO_MOVE": "0",
            "RESALE": "0",
            "ADDRESS": "Bandra, Mumbai",
            "LONGITUDE": "72.84",
            "LATITUDE": "19.05",
            TARGET_COLUMN: "250.0",
        },
        {
            "POSTED_BY": "Builder",
            "UNDER_CONSTRUCTION": "1",
            "RERA": "1",
            "BHK_NO.": "1",
            "BHK_OR_RK": "RK",
            "SQUARE_FT": "450.0",
            "READY_TO_MOVE": "0",
            "RESALE": "0",
            "ADDRESS": "Sector-137, Noida",
            "LONGITUDE": "77.40",
            "LATITUDE": "28.50",
            TARGET_COLUMN: "35.0",
        },
    ]


# ==================== Preprocessing & Schema Tests ====================

def test_extract_city_from_address():
    """Verify city parsing rule across standard, multi-comma, and edge-case addresses."""
    # Standard format: Locality, City
    assert extract_city_from_address("Whitefield, Bangalore") == "Bangalore"
    assert extract_city_from_address("Bandra (West), Mumbai") == "Mumbai"

    # Multi-comma format: Locality, Suburb, City
    assert (
        extract_city_from_address("Suddaguntepalya, C V Raman Nagar, Bangalore")
        == "Bangalore"
    )

    # Whitespace trimming
    assert extract_city_from_address("  Koramangala ,  Bangalore  ") == "Bangalore"

    # Single token (no comma)
    assert extract_city_from_address("Kolkata") == "Kolkata"

    # Blank / None fallbacks
    assert extract_city_from_address("") == "Unknown"
    assert extract_city_from_address(None) == "Unknown"


def test_city_extractor_frequency_and_unseen_handling():
    """Verify CityExtractor keeps frequent cities and maps infrequent/unseen cities to 'Other'."""
    extractor = CityExtractor(min_frequency=2, other_label="Other")

    # Training partition fixture: Bangalore appears 2x, Mumbai 2x, Shimla 1x
    train_records = [
        {"ADDRESS": "Locality 1, Bangalore"},
        {"ADDRESS": "Locality 2, Bangalore"},
        {"ADDRESS": "Locality 1, Mumbai"},
        {"ADDRESS": "Locality 2, Mumbai"},
        {"ADDRESS": "Locality 1, Shimla"},  # Below threshold
    ]

    extractor.fit(train_records)

    assert "Bangalore" in extractor.known_cities_
    assert "Mumbai" in extractor.known_cities_
    assert "Shimla" not in extractor.known_cities_

    # Transform known city
    res_known = extractor.transform({"ADDRESS": "New Locality, Bangalore"})
    assert res_known["CITY"] == "Bangalore"

    # Transform infrequent city seen in training
    res_infreq = extractor.transform({"ADDRESS": "Mall Road, Shimla"})
    assert res_infreq["CITY"] == "Other"

    # Transform completely unseen city (e.g. from validation or inference)
    res_unseen = extractor.transform({"ADDRESS": "Civil Lines, Nagpur"})
    assert res_unseen["CITY"] == "Other"


def test_collinear_feature_dropper(sample_valid_property_dict):
    """Verify READY_TO_MOVE is cleanly dropped to eliminate perfect multicollinearity."""
    dropper = CollinearFeatureDropper()
    transformed = dropper.transform(sample_valid_property_dict)

    assert COLLINEAR_DROPPED_COLUMN not in transformed
    assert "UNDER_CONSTRUCTION" in transformed
    assert transformed["UNDER_CONSTRUCTION"] == 0


def test_coordinate_validation_bounds():
    """Verify coordinates outside India bounds are detected and set to NaN."""
    # Valid coordinates (Bangalore)
    assert is_coordinate_in_india(12.97, 77.59) is True

    # Invalid coordinates (Reversed or outside bounding box)
    assert is_coordinate_in_india(-37.7, 59.9) is False
    assert is_coordinate_in_india(12.97, 152.9) is False
    assert is_coordinate_in_india(None, 77.59) is False

    validator = CoordinateValidator()
    valid_dict = {"LATITUDE": 12.97, "LONGITUDE": 77.59}
    invalid_dict = {"LATITUDE": 55.0, "LONGITUDE": 120.0}

    out_valid = validator.transform(valid_dict)
    assert out_valid["LATITUDE"] == 12.97
    assert out_valid["LONGITUDE"] == 77.59

    out_invalid = validator.transform(invalid_dict)
    assert np.isnan(out_invalid["LATITUDE"])
    assert np.isnan(out_invalid["LONGITUDE"])


def test_target_leakage_guard():
    """Verify TargetLeakageGuard raises ValueError when TARGET column is present in feature matrix."""
    guard = TargetLeakageGuard()

    # Leaked input (contains TARGET)
    leaked_features = {
        "SQUARE_FT": 1200.0,
        "BHK_NO.": 2,
        TARGET_COLUMN: 55.0,
    }

    with pytest.raises(ValueError, match="Target leakage detected"):
        guard.transform(leaked_features)

    # Safe input (no TARGET)
    safe_features = {
        "SQUARE_FT": 1200.0,
        "BHK_NO.": 2,
    }
    # Should execute without error
    out = guard.transform(safe_features)
    assert TARGET_COLUMN not in out


def test_separate_features_and_target_records(sample_property_records):
    """Verify strict separation of target from features with no residual leakage."""
    X_records, y_values = separate_features_and_target_records(sample_property_records)

    assert len(X_records) == 3
    assert len(y_values) == 3
    assert y_values == [65.0, 250.0, 35.0]

    for record in X_records:
        assert TARGET_COLUMN not in record


def test_train_val_split_indices_reproducibility():
    """Verify reproducible train/validation partition generation with zero index overlap."""
    n_samples = 100
    train_idx1, val_idx1 = train_val_split_indices(n_samples, val_size=0.2, random_state=42)
    train_idx2, val_idx2 = train_val_split_indices(n_samples, val_size=0.2, random_state=42)

    # Strict reproducibility
    assert train_idx1 == train_idx2
    assert val_idx1 == val_idx2

    # Partition sizes
    assert len(val_idx1) == 20
    assert len(train_idx1) == 80

    # Zero overlap (leakage prevention)
    overlap = set(train_idx1).intersection(set(val_idx1))
    assert len(overlap) == 0


def test_pydantic_property_input_schema_valid(sample_valid_property_dict):
    """Verify Pydantic input schema validates conformant property attributes."""
    schema = PropertyInputSchema(**sample_valid_property_dict)
    assert schema.POSTED_BY == "Owner"
    assert schema.BHK_NO == 2
    assert schema.SQUARE_FT == 1250.0
    assert schema.ADDRESS == "Whitefield, Bangalore"


def test_pydantic_property_input_schema_invalid():
    """Verify Pydantic input schema rejects non-conforming inputs with ValidationError."""
    # Negative square footage
    with pytest.raises(ValidationError):
        PropertyInputSchema(
            POSTED_BY="Owner",
            UNDER_CONSTRUCTION=0,
            RERA=1,
            **{"BHK_NO.": 2},
            BHK_OR_RK="BHK",
            SQUARE_FT=-500.0,
            RESALE=1,
            ADDRESS="Whitefield, Bangalore",
        )

    # Empty address
    with pytest.raises(ValidationError):
        PropertyInputSchema(
            POSTED_BY="Owner",
            UNDER_CONSTRUCTION=0,
            RERA=1,
            **{"BHK_NO.": 2},
            BHK_OR_RK="BHK",
            SQUARE_FT=1000.0,
            RESALE=1,
            ADDRESS="   ",
        )

    # Invalid category
    with pytest.raises(ValidationError):
        PropertyInputSchema(
            POSTED_BY="UnregisteredAgency",  # Invalid
            UNDER_CONSTRUCTION=0,
            RERA=1,
            **{"BHK_NO.": 2},
            BHK_OR_RK="BHK",
            SQUARE_FT=1000.0,
            RESALE=1,
            ADDRESS="Whitefield, Bangalore",
        )


def test_prediction_response_schema():
    """Verify prediction response schema enforces mandatory disclaimer and non-negative price."""
    resp = PredictionResponseSchema(
        estimated_price_lacs=62.5,
        formatted_price="₹62.50 Lakhs",
        model_version="EstateIQ-v1.0-RF",
    )
    assert resp.estimated_price_lacs == 62.5
    assert "educational and decision-support" in resp.disclaimer


def test_full_pipeline_fit_transform_pandas(sample_property_records):
    """Verify build_feature_preprocessor() fits on training records and transforms without error."""
    import pandas as pd
    from ml.pipeline.preprocessing import build_feature_preprocessor

    df = pd.DataFrame(sample_property_records).drop(columns=[TARGET_COLUMN])
    pipeline = build_feature_preprocessor(city_min_frequency=1)

    # Fit and transform
    X_proc = pipeline.fit_transform(df)
    assert isinstance(X_proc, np.ndarray)
    assert X_proc.shape[0] == len(sample_property_records)
    # Check no NaNs remain after imputation
    assert not np.isnan(X_proc).any()


def test_data_loader_prohibits_test_file():
    """Verify load_and_split_training_data refuses test.csv to prevent label-leakage violations."""
    from ml.pipeline.data_loader import load_and_split_training_data

    with pytest.raises(ValueError, match="Prohibited dataset"):
        load_and_split_training_data(data_path="ml/training/data/raw/original/test.csv")

