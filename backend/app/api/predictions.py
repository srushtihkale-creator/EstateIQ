"""Predictions API Router for Inference Execution and Isolated User History."""

from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.app.core.rate_limit import check_prediction_rate_limit
from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.schemas.prediction import (
    PropertyInputSchema,
    PredictionResponse,
    PredictionHistoryItem,
    PredictionHistoryPage,
)
from backend.app.services.prediction_service import PredictionService
from backend.app.api.deps import get_current_user, get_optional_user

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_prediction_rate_limit)],
    summary="Generate price estimate for property",
)
def predict_property_price(
    payload: PropertyInputSchema,
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """Run model inference on property attributes.

    If the request is from an authenticated user, the prediction is saved to their
    private history. Guests receive the prediction without database persistence.
    """
    # Convert validated pydantic payload to dictionary format expected by preprocessor
    property_dict = payload.model_dump(by_alias=True)

    prediction = PredictionService.predict_price(
        property_data=property_dict,
        user=current_user,
        db=db,
    )

    return prediction


@router.get(
    "",
    response_model=PredictionHistoryPage,
    summary="Get authenticated user's prediction history",
)
def get_prediction_history(
    page: int = Query(default=1, ge=1, description="Page number"),
    size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve paginated prediction records belonging strictly to the current user."""
    return PredictionService.get_user_history(
        db=db,
        user_id=current_user.id,
        page=page,
        size=size,
    )


@router.get(
    "/{prediction_id}",
    response_model=PredictionHistoryItem,
    summary="Get single prediction by ID",
)
def get_single_prediction(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve details of a specific prediction record, strictly enforcing ownership."""
    return PredictionService.get_user_prediction_by_id(
        db=db,
        user_id=current_user.id,
        prediction_id=prediction_id,
    )
