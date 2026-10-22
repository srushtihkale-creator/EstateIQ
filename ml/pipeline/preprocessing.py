"""Data Preprocessing Pipeline for EstateIQ.

Implements reusable, leakage-safe data transformation using Scikit-Learn
Pipeline and ColumnTransformer standards.
"""

from typing import List, Optional, Set, Dict, Any, Union
import numpy as np

from ml.common.constants import (
    TARGET_COLUMN,
    COLLINEAR_DROPPED_COLUMN,
    INDIA_BOUNDING_BOX,
    CITY_MIN_FREQUENCY,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    BINARY_COLUMNS,
)

# Optional scikit-learn imports with graceful fallback reporting
try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, FunctionTransformer
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Minimal stub classes so class definitions do not fail on import
    class BaseEstimator:
        pass
    class TransformerMixin:
        pass


def extract_city_from_address(address: str) -> str:
    """Extract standardized city name from an address string.

    Rule:
    - Split address by comma (',').
    - Take the final non-empty token, stripped of whitespace.
    - If no comma is present or address is empty, return trimmed string or 'Unknown'.
    """
    if not address or not isinstance(address, str):
        return "Unknown"
    tokens = [t.strip() for t in address.split(",") if t.strip()]
    if not tokens:
        return "Unknown"
    return tokens[-1]


def is_coordinate_in_india(lat: Optional[float], lon: Optional[float]) -> bool:
    """Validate whether latitude and longitude lie within the geographic India bounding box."""
    if lat is None or lon is None:
        return False
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (ValueError, TypeError):
        return False

    if np.isnan(lat_f) or np.isnan(lon_f):
        return False

    lat_ok = INDIA_BOUNDING_BOX["lat_min"] <= lat_f <= INDIA_BOUNDING_BOX["lat_max"]
    lon_ok = INDIA_BOUNDING_BOX["lon_min"] <= lon_f <= INDIA_BOUNDING_BOX["lon_max"]
    return lat_ok and lon_ok


def to_float_array(x):
    """Convert input array to float dtype for estimator compatibility (top-level for picklability)."""
    return np.asarray(x, dtype=float)


class TargetLeakageGuard(BaseEstimator, TransformerMixin):
    """Transformer ensuring target column cannot enter the feature matrix."""

    def __init__(self, target_column: str = TARGET_COLUMN):
        self.target_column = target_column

    def fit(self, X, y=None):
        self._check_leakage(X)
        return self

    def transform(self, X):
        self._check_leakage(X)
        return X

    def _check_leakage(self, X):
        cols = []
        if hasattr(X, "columns"):
            cols = list(X.columns)
        elif isinstance(X, dict):
            cols = list(X.keys())
        if self.target_column in cols:
            raise ValueError(
                f"Target leakage detected! '{self.target_column}' is present in the feature matrix. "
                "Target must be separated explicitly from features before preprocessing."
            )


class CollinearFeatureDropper(BaseEstimator, TransformerMixin):
    """Transformer to drop READY_TO_MOVE due to 100% exact collinearity with UNDER_CONSTRUCTION.

    Data audit finding:
    READY_TO_MOVE + UNDER_CONSTRUCTION == 1 across all 29,050 records.
    Dropping READY_TO_MOVE prevents rank deficiency in linear models.
    """

    def __init__(self, drop_column: str = COLLINEAR_DROPPED_COLUMN):
        self.drop_column = drop_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "drop") and hasattr(X, "columns"):
            if self.drop_column in X.columns:
                return X.drop(columns=[self.drop_column])
            return X.copy()
        elif isinstance(X, dict):
            res = dict(X)
            res.pop(self.drop_column, None)
            return res
        return X


class CoordinateValidator(BaseEstimator, TransformerMixin):
    """Transformer validating coordinates against India bounds, replacing invalid pairs with NaN.

    Downstream SimpleImputer fits median coordinates strictly on training data.
    """

    def __init__(self, lat_col: str = "LATITUDE", lon_col: str = "LONGITUDE"):
        self.lat_col = lat_col
        self.lon_col = lon_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "copy") and hasattr(X, "columns"):
            X_out = X.copy()
            if self.lat_col in X_out.columns and self.lon_col in X_out.columns:
                lats = X_out[self.lat_col].astype(float)
                lons = X_out[self.lon_col].astype(float)
                invalid_mask = (
                    (lats < INDIA_BOUNDING_BOX["lat_min"])
                    | (lats > INDIA_BOUNDING_BOX["lat_max"])
                    | (lons < INDIA_BOUNDING_BOX["lon_min"])
                    | (lons > INDIA_BOUNDING_BOX["lon_max"])
                )
                X_out.loc[invalid_mask, self.lat_col] = np.nan
                X_out.loc[invalid_mask, self.lon_col] = np.nan
            return X_out
        elif isinstance(X, dict):
            res = dict(X)
            lat = res.get(self.lat_col)
            lon = res.get(self.lon_col)
            if lat is not None and lon is not None:
                if not is_coordinate_in_india(lat, lon):
                    res[self.lat_col] = np.nan
                    res[self.lon_col] = np.nan
            return res
        return X


class CityExtractor(BaseEstimator, TransformerMixin):
    """Transformer extracting City from ADDRESS and grouping low-frequency cities into 'Other'.

    Fit exclusively on the training partition to prevent validation/test leakage.
    Unseen cities in validation or inference are safely assigned to 'Other'.
    """

    def __init__(
        self,
        address_col: str = "ADDRESS",
        city_col: str = "CITY",
        min_frequency: int = CITY_MIN_FREQUENCY,
        other_label: str = "Other",
    ):
        self.address_col = address_col
        self.city_col = city_col
        self.min_frequency = min_frequency
        self.other_label = other_label
        self.known_cities_: Set[str] = set()

    def fit(self, X, y=None):
        addresses = []
        if hasattr(X, "columns") and self.address_col in X.columns:
            addresses = X[self.address_col].dropna().astype(str).tolist()
        elif isinstance(X, dict):
            val = X.get(self.address_col, "")
            if val:
                addresses = [str(val)]
        elif isinstance(X, (list, tuple)):
            addresses = [str(item.get(self.address_col, "")) for item in X]

        city_counts: Dict[str, int] = {}
        for addr in addresses:
            c = extract_city_from_address(addr)
            city_counts[c] = city_counts.get(c, 0) + 1

        self.known_cities_ = {
            c for c, count in city_counts.items() if count >= self.min_frequency
        }
        return self

    def transform(self, X):
        if hasattr(X, "copy") and hasattr(X, "columns"):
            X_out = X.copy()
            if self.address_col in X_out.columns:
                extracted = X_out[self.address_col].astype(str).apply(extract_city_from_address)
                X_out[self.city_col] = extracted.apply(
                    lambda c: c if c in self.known_cities_ else self.other_label
                )
                X_out = X_out.drop(columns=[self.address_col])
            return X_out
        elif isinstance(X, dict):
            res = dict(X)
            raw_addr = str(res.pop(self.address_col, ""))
            c = extract_city_from_address(raw_addr)
            res[self.city_col] = c if c in self.known_cities_ else self.other_label
            return res
        return X


def build_feature_preprocessor(
    numeric_scaler: str = "robust",
    city_min_frequency: int = CITY_MIN_FREQUENCY,
) -> "Pipeline":
    """Construct a full Scikit-Learn feature preprocessing pipeline.

    The pipeline handles:
    1. Guarding against target leakage.
    2. Dropping collinear feature READY_TO_MOVE.
    3. Validating and masking out-of-bounds geographic coordinates.
    4. Extracting and thresholding city names from ADDRESS.
    5. Imputing numeric features (median) + robust scaling.
    6. One-hot encoding categorical features with handle_unknown='ignore'.
    7. Passing binary indicators through unchanged.

    Note: Every component is strictly fitted during pipeline.fit(X_train).
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is not currently installed. "
            "Please install scikit-learn, pandas, and joblib to execute this pipeline."
        )

    # Scaler selection: RobustScaler is preferred due to extreme area outliers (up to 254M sq ft)
    scaler = RobustScaler() if numeric_scaler == "robust" else StandardScaler()

    numeric_transformer = Pipeline(
        steps=[
            ("to_float", FunctionTransformer(to_float_array)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_float", FunctionTransformer(to_float_array)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
            ("bin", binary_transformer, BINARY_COLUMNS),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("leakage_guard", TargetLeakageGuard()),
            ("collinear_dropper", CollinearFeatureDropper()),
            ("coord_validator", CoordinateValidator()),
            ("city_extractor", CityExtractor(min_frequency=city_min_frequency)),
            ("col_transformer", column_transformer),
        ]
    )

    return pipeline
