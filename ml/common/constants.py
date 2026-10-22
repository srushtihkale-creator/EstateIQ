"""Constants and configuration settings for EstateIQ ML pipeline."""

# Target column definition
TARGET_COLUMN = "TARGET(PRICE_IN_LACS)"

# Raw columns expected in the dataset
RAW_FEATURE_COLUMNS = [
    "POSTED_BY",
    "UNDER_CONSTRUCTION",
    "RERA",
    "BHK_NO.",
    "BHK_OR_RK",
    "SQUARE_FT",
    "READY_TO_MOVE",
    "RESALE",
    "ADDRESS",
    "LONGITUDE",
    "LATITUDE",
]

# Feature columns after dropping exact collinear feature (READY_TO_MOVE = 1 - UNDER_CONSTRUCTION)
FEATURE_COLUMNS = [
    "POSTED_BY",
    "UNDER_CONSTRUCTION",
    "RERA",
    "BHK_NO.",
    "BHK_OR_RK",
    "SQUARE_FT",
    "RESALE",
    "ADDRESS",
    "LONGITUDE",
    "LATITUDE",
]

# Redundant feature identified in data audit
COLLINEAR_DROPPED_COLUMN = "READY_TO_MOVE"

# Categorical features
CATEGORICAL_COLUMNS = ["POSTED_BY", "BHK_OR_RK", "CITY"]

# Numeric features requiring imputation and scaling
NUMERIC_COLUMNS = ["BHK_NO.", "SQUARE_FT", "LONGITUDE", "LATITUDE"]

# Binary indicator features (already 0 or 1)
BINARY_COLUMNS = ["UNDER_CONSTRUCTION", "RERA", "RESALE"]

# Geographic bounding box for India coordinates
# Latitude: 6.5° N to 37.5° N
# Longitude: 68.0° E to 97.5° E
INDIA_BOUNDING_BOX = {
    "lat_min": 6.5,
    "lat_max": 37.5,
    "lon_min": 68.0,
    "lon_max": 97.5,
}

# Minimum frequency for city categorical encoding (cities below this threshold mapped to 'Other')
CITY_MIN_FREQUENCY = 50

# Default reproducible split parameters
DEFAULT_RANDOM_STATE = 42
DEFAULT_VAL_SIZE = 0.20

# Standard dataset paths
DEFAULT_TRAIN_CLEANED_PATH = "ml/training/data/processed/train_cleaned.csv"
DEFAULT_TEST_CLEANED_PATH = "ml/training/data/processed/test_cleaned.csv"
DEFAULT_RAW_TRAIN_PATH = "ml/training/data/raw/original/train.csv"
DEFAULT_RAW_TEST_PATH = "ml/training/data/raw/original/test.csv"
