"""Pydantic schemas and data validation models for EstateIQ ML pipeline."""
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class PropertyInputSchema(BaseModel):
    """Schema validating raw input for a property price prediction request."""

    POSTED_BY: Literal["Owner", "Dealer", "Builder"] = Field(
        ...,
        description="Type of party who listed the property (Owner, Dealer, or Builder)",
        examples=["Owner"],
    )
    UNDER_CONSTRUCTION: Literal[0, 1] = Field(
        ...,
        description="Whether property is under construction (1) or completed (0)",
        examples=[0],
    )
    RERA: Literal[0, 1] = Field(
        ...,
        description="Whether property is RERA-approved (1) or not (0)",
        examples=[1],
    )
    BHK_NO: int = Field(
        ...,
        alias="BHK_NO.",
        ge=1,
        le=20,
        description="Number of bedrooms (1 to 20)",
        examples=[2],
    )
    BHK_OR_RK: Literal["BHK", "RK"] = Field(
        ...,
        description="Unit type: BHK (Bedroom Hall Kitchen) or RK (Room Kitchen)",
        examples=["BHK"],
    )
    SQUARE_FT: float = Field(
        ...,
        gt=0.0,
        le=100000.0,
        description="Total built-up/carpet area in square feet",
        examples=[1200.0],
    )
    RESALE: Literal[0, 1] = Field(
        ...,
        description="Whether property is a resale unit (1) or new sale (0)",
        examples=[1],
    )
    ADDRESS: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description="Full address or locality with city (e.g., 'Whitefield, Bangalore')",
        examples=["Whitefield, Bangalore"],
    )
    LONGITUDE: Optional[float] = Field(
        None,
        ge=68.0,
        le=97.5,
        description="Longitude in decimal degrees (India range: 68.0 to 97.5)",
        examples=[77.59796],
    )
    LATITUDE: Optional[float] = Field(
        None,
        ge=6.5,
        le=37.5,
        description="Latitude in decimal degrees (India range: 6.5 to 37.5)",
        examples=[12.96991],
    )
    # Optional field in case raw legacy input includes READY_TO_MOVE
    READY_TO_MOVE: Optional[Literal[0, 1]] = Field(
        None,
        description="Optional legacy flag; inverse of UNDER_CONSTRUCTION. Dropped in pipeline.",
    )

    @field_validator("ADDRESS")
    @classmethod
    def validate_address_not_blank(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("ADDRESS cannot be empty or pure whitespace.")
        return clean

    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


class PredictionResponseSchema(BaseModel):
    """Schema for prediction endpoint response."""

    estimated_price_lacs: float = Field(
        ...,
        ge=0.0,
        description="Estimated price in Indian Lakhs (1 Lac = 100,000 INR)",
    )
    formatted_price: str = Field(
        ...,
        description="Human-readable formatted price (e.g., '₹62.50 Lakhs' or '₹1.25 Crore')",
    )
    price_unit: str = Field(
        default="Lakhs INR",
        description="Currency unit definition",
    )
    model_version: str = Field(
        ...,
        description="Version/identifier of the trained model that generated this estimate",
    )
    disclaimer: str = Field(
        default=(
            "EstateIQ estimates are statistical model outputs for educational and decision-support "
            "purposes. They are not official valuations, appraisals, or guaranteed transaction prices."
        ),
        description="Mandatory disclaimer as required by PRD and SRS.",
    )
