"""Pydantic Request and Response Schemas for User Accounts and Sessions."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserRegisterRequest(BaseModel):
    """Schema validating user registration requests."""

    email: EmailStr = Field(..., description="Valid user email address", examples=["user@example.com"])
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password (minimum 8 characters)",
        examples=["SecurePassw0rd!"],
    )
    full_name: Optional[str] = Field(None, max_length=100, description="Optional user full name")


class UserLoginRequest(BaseModel):
    """Schema validating login credentials."""

    email: EmailStr = Field(..., description="Registered email address")
    password: str = Field(..., min_length=1, max_length=128, description="User password")


class UserResponse(BaseModel):
    """Safe public representation of user entity (excludes password hash)."""

    id: int
    email: EmailStr
    full_name: Optional[str] = None
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    """Response returned upon successful login or registration."""

    user: UserResponse
    csrf_token: str = Field(..., description="CSRF token for state-changing requests")
    message: str = "Authentication successful"
