"""Health and Readiness Probes for EstateIQ Backend."""

import os
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.session import get_db
from backend.app.services.prediction_service import PredictionService

router = APIRouter(tags=["Health & Diagnostics"])


@router.get("/health", summary="Service liveness health check")
def health_check(db: Session = Depends(get_db)):
    """Check process health and database connectivity without leaking environment secrets."""
    db_status = "unknown"
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return {
        "status": "ok" if db_status == "connected" else "degraded",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "database": db_status,
    }


@router.get("/ready", summary="Service readiness probe")
def readiness_check(db: Session = Depends(get_db)):
    """Verify readiness: ensures database connectivity and model artifact loading."""
    # 1. Database check
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database readiness check failed. Unable to establish database connection.",
        )

    # 2. Model pipeline artifact check
    if not os.path.exists(settings.MODEL_ARTIFACT_PATH):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact file is not present on disk.",
        )

    try:
        pipeline = PredictionService.get_pipeline()
        model_name = pipeline.metadata.get("model_name", "DecisionTree_Raw")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model readiness check failed: {str(e)}",
        )

    return {
        "status": "ready",
        "database": "connected",
        "model_artifact": "loaded",
        "model_name": model_name,
    }
