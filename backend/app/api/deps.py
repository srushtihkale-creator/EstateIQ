"""FastAPI Dependencies for Authentication, CSRF Protection, and Database Sessions."""

from typing import Optional, Tuple
from fastapi import Depends, Request, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.security import verify_csrf_token
from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.models.session import UserSession
from backend.app.services.auth_service import AuthService


def get_current_user_and_session(
    request: Request,
    db: Session = Depends(get_db),
) -> Tuple[User, UserSession]:
    """Extract and validate active session from HttpOnly cookie.

    Raises 401 Unauthorized if cookie is missing, expired, or invalid.
    """
    session_id = request.cookies.get(settings.SESSION_COOKIE_NAME)
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
        )

    user, user_session = AuthService.get_user_and_session_by_token(db, session_id)
    if not user or not user_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired or is invalid. Please sign in again.",
        )

    return user, user_session


def get_current_user(
    user_and_session: Tuple[User, UserSession] = Depends(get_current_user_and_session),
) -> User:
    """Dependency returning the active authenticated User."""
    return user_and_session[0]


def get_optional_user(
    request: Request,
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Dependency returning User if active session cookie is present, or None if guest."""
    session_id = request.cookies.get(settings.SESSION_COOKIE_NAME)
    if not session_id:
        return None

    user, _ = AuthService.get_user_and_session_by_token(db, session_id)
    return user


def verify_csrf_protection(
    request: Request,
    user_and_session: Tuple[User, UserSession] = Depends(get_current_user_and_session),
) -> None:
    """Validate CSRF token on state-changing authenticated requests (POST, PUT, DELETE, PATCH).

    Token must be provided in the X-CSRF-Token header and match the active session.
    """
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        header_token = request.headers.get(settings.CSRF_HEADER_NAME)
        _, user_session = user_and_session

        # Match header against the session record's CSRF token
        if not header_token or not verify_csrf_token(header_token, user_session.csrf_token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF verification failed. Missing or invalid X-CSRF-Token header.",
            )
