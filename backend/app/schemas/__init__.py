"""EstateIQ Pydantic Schemas Package."""
from backend.app.schemas.user import UserRegisterRequest, UserLoginRequest, UserResponse, AuthResponse
from backend.app.schemas.prediction import (
    PropertyInputSchema,
    PredictionResponse,
    PredictionHistoryItem,
    PredictionHistoryPage,
)
from backend.app.schemas.model import ModelMetricsResponse

__all__ = [
    "UserRegisterRequest",
    "UserLoginRequest",
    "UserResponse",
    "AuthResponse",
    "PropertyInputSchema",
    "PredictionResponse",
    "PredictionHistoryItem",
    "PredictionHistoryPage",
    "ModelMetricsResponse",
]
