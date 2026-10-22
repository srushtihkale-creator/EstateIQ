"""Prediction History Model for User-Isolated Prediction Tracking."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from backend.app.db.session import Base


class PredictionHistory(Base):
    """Historical property prediction entity isolated by owning user."""

    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    inputs = Column(Text, nullable=False)  # JSON-serialized input feature snapshot
    predicted_price_lacs = Column(Float, nullable=False)
    currency_unit = Column(String(50), default="Lakhs INR", nullable=False)
    model_version = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True, nullable=False)

    # Relationships
    user = relationship("User", back_populates="predictions")
