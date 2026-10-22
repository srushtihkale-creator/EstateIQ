"""Prediction and Inference Execution Service."""

import json
import math
import time
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
import numpy as np
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.models.user import User
from backend.app.models.prediction import PredictionHistory
from backend.app.schemas.prediction import (
    PredictionResponse,
    PredictionHistoryItem,
    PredictionHistoryPage,
)
from ml.pipeline.artifact import (
    EstateIQInferencePipeline,
    load_production_pipeline_artifact,
)

# In-memory singleton container for the loaded model pipeline
_cached_pipeline: Optional[EstateIQInferencePipeline] = None


def format_inr_price(price_in_lacs: float) -> str:
    """Format numeric price in Lakhs to human-friendly Indian Rupee representation.

    Rules:
    - 1 Lac = 100,000 INR
    - >= 100 Lacs (>= 10,000,000 INR) is represented in Crores (e.g., ₹1.25 Crore)
    - < 100 Lacs is represented in Lakhs (e.g., ₹62.50 Lakhs)
    """
    if price_in_lacs >= 100.0:
        crores = price_in_lacs / 100.0
        return f"₹{crores:.2f} Crore"
    return f"₹{price_in_lacs:.2f} Lakhs"


class PredictionService:
    """Service orchestrating ML inference, output validation, and isolated history persistence."""

    @staticmethod
    def get_pipeline() -> EstateIQInferencePipeline:
        """Retrieve the cached inference pipeline or load it from disk safely."""
        global _cached_pipeline
        if _cached_pipeline is not None:
            return _cached_pipeline

        artifact_path = settings.MODEL_ARTIFACT_PATH
        if not os.path.exists(artifact_path):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference model pipeline is not currently loaded or artifact is missing.",
            )

        try:
            _cached_pipeline = load_production_pipeline_artifact(artifact_path)
            return _cached_pipeline
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to initialize model inference engine: {str(e)}",
            )

    @staticmethod
    def predict_price(
        property_data: Dict[str, Any],
        user: Optional[User] = None,
        db: Optional[Session] = None,
    ) -> PredictionResponse:
        """Execute inference on validated property attributes and optionally persist result."""
        pipeline = PredictionService.get_pipeline()

        # Run inference through fitted preprocessor and estimator
        try:
            predictions = pipeline.predict(property_data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Inference execution error: {str(e)}",
            )

        if len(predictions) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Inference pipeline returned empty prediction output.",
            )

        raw_price = float(predictions[0])

        # Robustness validation: reject non-finite predictions
        if np.isnan(raw_price) or np.isinf(raw_price):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model generated a non-finite numerical prediction.",
            )

        # Enforce non-negative price
        price_lacs = max(0.0, round(raw_price, 2))
        formatted = format_inr_price(price_lacs)
        model_name = pipeline.metadata.get("model_name", "EstateIQ-DecisionTree")
        timestamp_str = datetime.now(timezone.utc).isoformat()

        prediction_id: Optional[int] = None

        # If user is authenticated and db session is provided, persist to private history
        if user and db:
            history_entry = PredictionHistory(
                user_id=user.id,
                inputs=json.dumps(property_data),
                predicted_price_lacs=price_lacs,
                currency_unit="Lakhs INR",
                model_version=model_name,
                created_at=datetime.now(timezone.utc),
            )
            db.add(history_entry)
            db.commit()
            db.refresh(history_entry)
            prediction_id = history_entry.id

        return PredictionResponse(
            prediction_id=prediction_id,
            estimated_price_lacs=price_lacs,
            formatted_price=formatted,
            price_unit="Lakhs INR",
            model_version=model_name,
            timestamp=timestamp_str,
        )

    @staticmethod
    def get_user_history(
        db: Session,
        user_id: int,
        page: int = 1,
        size: int = 20,
    ) -> PredictionHistoryPage:
        """Fetch user-isolated prediction history with pagination."""
        if page < 1:
            page = 1
        if size < 1 or size > 100:
            size = 20

        # Strict tenant query: enforce user_id ownership filter
        query = db.query(PredictionHistory).filter(PredictionHistory.user_id == user_id)
        total = query.count()

        records = (
            query.order_by(PredictionHistory.created_at.desc())
            .offset((page - 1) * size)
            .limit(size)
            .all()
        )

        items = []
        for r in records:
            try:
                inputs_dict = json.loads(r.inputs)
            except Exception:
                inputs_dict = {}

            items.append(
                PredictionHistoryItem(
                    id=r.id,
                    inputs=inputs_dict,
                    predicted_price_lacs=r.predicted_price_lacs,
                    formatted_price=format_inr_price(r.predicted_price_lacs),
                    currency_unit=r.currency_unit,
                    model_version=r.model_version,
                    created_at=r.created_at,
                )
            )

        pages = math.ceil(total / size) if total > 0 else 1

        return PredictionHistoryPage(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages,
        )

    @staticmethod
    def get_user_prediction_by_id(
        db: Session,
        user_id: int,
        prediction_id: int,
    ) -> PredictionHistoryItem:
        """Fetch a specific prediction record, strictly enforcing user ownership."""
        record = (
            db.query(PredictionHistory)
            .filter(
                PredictionHistory.id == prediction_id,
                PredictionHistory.user_id == user_id,  # Mandatory ownership check
            )
            .first()
        )

        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prediction record not found or access denied.",
            )

        try:
            inputs_dict = json.loads(record.inputs)
        except Exception:
            inputs_dict = {}

        return PredictionHistoryItem(
            id=record.id,
            inputs=inputs_dict,
            predicted_price_lacs=record.predicted_price_lacs,
            formatted_price=format_inr_price(record.predicted_price_lacs),
            currency_unit=record.currency_unit,
            model_version=record.model_version,
            created_at=record.created_at,
        )
