"""Pydantic Schemas for Property Predictions and Prediction History."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Re-use the audited and verified property input validation model
from ml.pipeline.schemas import PropertyInputSchema


class PredictionResponse(BaseModel):
    """Schema for prediction response payload."""

    prediction_id: Optional[int] = Field(
        None,
        description="ID of saved prediction history record if authenticated",
    )
    estimated_price_lacs: float = Field(
        ...,
        ge=0.0,
        description="Estimated price in Lakhs INR (1 Lac = 100,000 INR)",
    )
    formatted_price: str = Field(
        ...,
        description="Human-readable formatted price string (e.g., '₹62.50 Lakhs' or '₹1.25 Crore')",
    )
    price_unit: str = Field(
        default="Lakhs INR",
        description="Standardized price unit",
    )
    model_version: str = Field(
        ...,
        description="Version/identifier of model that generated the estimate",
    )
    disclaimer: str = Field(
        default=(
            "EstateIQ estimates are statistical machine-learning outputs for educational and decision-support "
            "purposes. They are not certified valuations, formal appraisals, or guaranteed transaction prices."
        ),
        description="Mandatory disclaimer required by PRD and SRS.",
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 prediction generation timestamp",
    )


class PredictionHistoryItem(BaseModel):
    """Individual prediction history record returned to authenticated user."""

    id: int
    inputs: Dict[str, Any]
    predicted_price_lacs: float
    formatted_price: str
    currency_unit: str
    model_version: str
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionHistoryPage(BaseModel):
    """Paginated collection of user's own prediction history."""

    items: List[PredictionHistoryItem]
    total: int
    page: int
    size: int
    pages: int
