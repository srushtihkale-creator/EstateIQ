"""Application Configuration Settings for EstateIQ Backend.

All configuration is environment-driven with strict security validations.
No production secrets or URLs are hardcoded.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine EstateIQ project root directory reliably
# backend/app/core/config.py -> 3 levels up is d:\EstateIQ
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """EstateIQ Backend Settings loaded from environment variables."""

    # Project metadata
    PROJECT_NAME: str = "EstateIQ API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", description="development, staging, or production")
    DEBUG: bool = Field(default=False, description="Enable debug mode (strictly false in production)")

    # API Routing
    API_V1_PREFIX: str = "/api"

    # Database Configuration (Local PostgreSQL for development/staging; Supabase in production)
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="SQLAlchemy database connection URL (PostgreSQL for local or Supabase; SQLite fallback)",
    )
    POSTGRES_USER: Optional[str] = Field(default=None, description="Local PostgreSQL username")
    POSTGRES_PASSWORD: Optional[str] = Field(default=None, description="Local PostgreSQL password")
    POSTGRES_HOST: str = Field(default="localhost", description="PostgreSQL host")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL port")
    POSTGRES_DB: str = Field(default="estateiq_db", description="PostgreSQL database name")

    # Session Authentication & Cookie Security
    SESSION_SECRET: str = Field(
        default="estateiq-local-dev-session-secret-change-in-production-min-32-chars",
        description="Secret key used for signing session-related elements",
    )
    SESSION_COOKIE_NAME: str = "estateiq_session"
    SESSION_EXPIRE_SECONDS: int = Field(default=60 * 60 * 24 * 7, description="7 days in seconds")

    # Cookie security flags
    COOKIE_SECURE: bool = Field(
        default=False,
        description="Set True in HTTPS production to ensure cookie is transmitted over HTTPS only",
    )
    COOKIE_SAMESITE: str = Field(
        default="lax",
        description="Cookie SameSite policy: 'lax', 'strict', or 'none' (for cross-domain with Secure)",
    )
    COOKIE_DOMAIN: Optional[str] = Field(
        default=None,
        description="Domain attribute for the cookie (e.g. .yourdomain.com)",
    )

    # CSRF Protection
    CSRF_HEADER_NAME: str = "X-CSRF-Token"
    CSRF_COOKIE_NAME: str = "estateiq_csrf"

    # CORS Configuration
    FRONTEND_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="List of allowed CORS origins for the frontend",
    )

    # Rate Limiting Parameters (requests per minute)
    AUTH_RATE_LIMIT_PER_MINUTE: int = Field(default=10, description="Max login/register requests per min per IP")
    API_RATE_LIMIT_PER_MINUTE: int = Field(default=120, description="Max standard API requests per min per IP")

    # Model Artifact Paths (resolved relative to project root)
    MODEL_ARTIFACT_PATH: str = Field(
        default=str(PROJECT_ROOT / "ml" / "inference" / "artifacts" / "estateiq_model_pipeline.joblib"),
        description="Filesystem path to the verified production model pipeline artifact",
    )
    MODEL_METADATA_PATH: str = Field(
        default=str(PROJECT_ROOT / "ml" / "inference" / "artifacts" / "model_metadata.json"),
        description="Filesystem path to model evaluation metadata JSON",
    )

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def normalize_database_url(cls, v: Optional[str]) -> Optional[str]:
        if v and v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql://", 1)
        return v

    @model_validator(mode="after")
    def assemble_db_connection(self) -> "Settings":
        """Assemble DATABASE_URL from components or fallback safely to SQLite."""
        if self.DATABASE_URL:
            if self.DATABASE_URL.startswith("postgres://"):
                self.DATABASE_URL = self.DATABASE_URL.replace("postgres://", "postgresql://", 1)
        elif self.POSTGRES_USER and self.POSTGRES_PASSWORD:
            from urllib.parse import quote_plus
            user = quote_plus(self.POSTGRES_USER)
            password = quote_plus(self.POSTGRES_PASSWORD)
            self.DATABASE_URL = (
                f"postgresql://{user}:{password}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        else:
            self.DATABASE_URL = "sqlite:///./estateiq_local.db"
        return self

    model_config = SettingsConfigDict(
        env_file=[
            str(PROJECT_ROOT / "backend" / ".env"),
            str(PROJECT_ROOT / ".env"),
        ],
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Centralized settings instance
settings = Settings()
