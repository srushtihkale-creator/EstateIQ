"""Authentication API Router for Registration, Login, Logout, and Profile."""

from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.rate_limit import check_auth_rate_limit
from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.schemas.user import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    AuthResponse,
)
from backend.app.services.auth_service import AuthService
from backend.app.api.deps import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _set_auth_cookies(response: Response, session_id: str, csrf_token: str) -> None:
    """Set secure HttpOnly session cookie and client-readable CSRF cookie."""
    # HttpOnly session cookie (unreadable by client JS to prevent XSS theft)
    response.set_cookie(
        key=settings.SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        max_age=settings.SESSION_EXPIRE_SECONDS,
        domain=settings.COOKIE_DOMAIN,
        path="/",
    )
    # CSRF cookie (readable by client to attach in X-CSRF-Token header)
    response.set_cookie(
        key=settings.CSRF_COOKIE_NAME,
        value=csrf_token,
        httponly=False,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        max_age=settings.SESSION_EXPIRE_SECONDS,
        domain=settings.COOKIE_DOMAIN,
        path="/",
    )


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_auth_rate_limit)],
    summary="Register a new user account",
)
def register(
    payload: UserRegisterRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """Create a new user account, hash the password, and establish an active session."""
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    user, session_id, csrf_token = AuthService.register_user(
        db=db,
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name,
        ip_address=client_ip,
        user_agent=user_agent,
    )

    _set_auth_cookies(response, session_id, csrf_token)

    return AuthResponse(
        user=UserResponse.model_validate(user),
        csrf_token=csrf_token,
        message="Account created successfully",
    )


@router.post(
    "/login",
    response_model=AuthResponse,
    dependencies=[Depends(check_auth_rate_limit)],
    summary="Sign in to user account",
)
def login(
    payload: UserLoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """Authenticate user credentials and establish a server-managed session."""
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    user, session_id, csrf_token = AuthService.authenticate_user(
        db=db,
        email=payload.email,
        password=payload.password,
        ip_address=client_ip,
        user_agent=user_agent,
    )

    _set_auth_cookies(response, session_id, csrf_token)

    return AuthResponse(
        user=UserResponse.model_validate(user),
        csrf_token=csrf_token,
        message="Signed in successfully",
    )


@router.post(
    "/logout",
    summary="Sign out and invalidate session",
)
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """Revoke session token on server and delete client authentication cookies."""
    session_id = request.cookies.get(settings.SESSION_COOKIE_NAME)
    AuthService.revoke_session(db, session_id)

    response.delete_cookie(
        key=settings.SESSION_COOKIE_NAME,
        domain=settings.COOKIE_DOMAIN,
        path="/",
    )
    response.delete_cookie(
        key=settings.CSRF_COOKIE_NAME,
        domain=settings.COOKIE_DOMAIN,
        path="/",
    )

    return {"message": "Signed out successfully"}


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    """Return the authenticated user's profile details."""
    return UserResponse.model_validate(current_user)
