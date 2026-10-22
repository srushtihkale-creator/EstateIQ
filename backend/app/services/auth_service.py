"""Authentication and Session Management Service."""

from datetime import datetime, timezone, timedelta
from typing import Tuple, Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.security import (
    hash_password,
    verify_password,
    needs_rehash,
    generate_session_id,
    hash_session_id,
    generate_csrf_token,
)
from backend.app.models.user import User
from backend.app.models.session import UserSession


class AuthService:
    """Service handling account registration, credential verification, and stateful sessions."""

    @staticmethod
    def register_user(
        db: Session,
        email: str,
        password: str,
        full_name: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Tuple[User, str, str]:
        """Register a new user account, hash password, and issue an active session."""
        normalized_email = email.lower().strip()

        # Check existing user
        existing = db.query(User).filter(User.email == normalized_email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An account with this email already exists.",
            )

        # Hash password securely
        hashed = hash_password(password)

        new_user = User(
            email=normalized_email,
            hashed_password=hashed,
            full_name=full_name.strip() if full_name else None,
            role="user",
            is_active=True,
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        # Issue initial session
        session_id, csrf_token = AuthService.create_user_session(
            db, new_user, ip_address=ip_address, user_agent=user_agent
        )

        return new_user, session_id, csrf_token

    @staticmethod
    def authenticate_user(
        db: Session,
        email: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Tuple[User, str, str]:
        """Verify user credentials and establish a new server-managed session."""
        normalized_email = email.lower().strip()
        user = db.query(User).filter(User.email == normalized_email).first()

        # Generic error message to prevent account enumeration
        invalid_credentials_exc = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

        if not user or not user.is_active:
            raise invalid_credentials_exc

        if not verify_password(password, user.hashed_password):
            raise invalid_credentials_exc

        # Safe migration: if hash is legacy (e.g. bcrypt) or needs rehash, upgrade to Argon2id
        if needs_rehash(user.hashed_password):
            user.hashed_password = hash_password(password)
            db.commit()
            db.refresh(user)

        session_id, csrf_token = AuthService.create_user_session(
            db, user, ip_address=ip_address, user_agent=user_agent
        )

        return user, session_id, csrf_token

    @staticmethod
    def create_user_session(
        db: Session,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Generate high-entropy session ID and CSRF token, storing only the hash in DB."""
        session_id = generate_session_id()
        session_hash = hash_session_id(session_id)
        csrf_token = generate_csrf_token()

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=settings.SESSION_EXPIRE_SECONDS)

        session_record = UserSession(
            user_id=user.id,
            session_id_hash=session_hash,
            csrf_token=csrf_token,
            ip_address=ip_address[:45] if ip_address else None,
            user_agent=user_agent[:255] if user_agent else None,
            expires_at=expires_at,
            revoked=False,
            created_at=now,
        )
        db.add(session_record)
        db.commit()

        return session_id, csrf_token

    @staticmethod
    def get_user_and_session_by_token(
        db: Session,
        session_id: Optional[str],
    ) -> Tuple[Optional[User], Optional[UserSession]]:
        """Validate session token, check expiry/revocation, and return active User & Session."""
        if not session_id:
            return None, None

        session_hash = hash_session_id(session_id)
        now = datetime.now(timezone.utc)

        user_session = (
            db.query(UserSession)
            .filter(
                UserSession.session_id_hash == session_hash,
                UserSession.revoked.is_(False),
                UserSession.expires_at > now,
            )
            .first()
        )

        if not user_session or not user_session.user or not user_session.user.is_active:
            return None, None

        return user_session.user, user_session

    @staticmethod
    def revoke_session(db: Session, session_id: Optional[str]) -> bool:
        """Revoke active user session upon logout."""
        if not session_id:
            return False

        session_hash = hash_session_id(session_id)
        session_record = db.query(UserSession).filter(UserSession.session_id_hash == session_hash).first()
        if session_record:
            session_record.revoked = True
            db.commit()
            return True
        return False
