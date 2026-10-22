"""Security, Cryptography, and Session Token Utilities for EstateIQ."""

import hashlib
import secrets
from typing import Optional

# Argon2id password hasher configuration (OWASP recommended parameters)
# Memory cost: 64MB (65536 KB), Time cost: 3 iterations, Parallelism: 4 threads
try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError, InvalidHash
    _argon2_hasher = PasswordHasher(
        time_cost=3,
        memory_cost=65536,
        parallelism=4,
        hash_len=32,
        salt_len=16,
    )
    ARGON2_AVAILABLE = True
except ImportError:
    ARGON2_AVAILABLE = False
    _argon2_hasher = None

# Optional passlib bcrypt context strictly for backward-compatible verification of legacy hashes
try:
    from passlib.context import CryptContext
    _bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    BCRYPT_AVAILABLE = True
except ImportError:
    _bcrypt_context = None
    BCRYPT_AVAILABLE = False


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password using Argon2id strictly.

    Argon2id is enforced as the required algorithm for all new passwords.
    Silent fallback to other algorithms is forbidden.
    """
    if not plain_password:
        raise ValueError("Password cannot be empty.")

    if not ARGON2_AVAILABLE or not _argon2_hasher:
        raise RuntimeError("Argon2 hashing is required but argon2-cffi is not installed.")

    return _argon2_hasher.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against the stored hash.

    Supports Argon2id natively, and preserves backward compatibility with legacy
    bcrypt or pbkdf2 hashes so that existing users can authenticate and be migrated.
    """
    if not plain_password or not hashed_password:
        return False

    try:
        if hashed_password.startswith("$argon2"):
            if ARGON2_AVAILABLE and _argon2_hasher:
                try:
                    return _argon2_hasher.verify(hashed_password, plain_password)
                except (VerifyMismatchError, InvalidHash):
                    return False
            return False

        if hashed_password.startswith("$2") or hashed_password.startswith("$bcrypt"):
            if BCRYPT_AVAILABLE and _bcrypt_context:
                return _bcrypt_context.verify(plain_password, hashed_password)
            return False

        if hashed_password.startswith("pbkdf2_sha256$"):
            _, salt, true_hash = hashed_password.split("$", 2)
            computed = hashlib.sha256((salt + plain_password).encode()).hexdigest()
            return secrets.compare_digest(computed, true_hash)

        return False
    except Exception:
        return False


def needs_rehash(hashed_password: str) -> bool:
    """Check if an existing password hash needs to be upgraded/rehashed to Argon2id."""
    if not hashed_password:
        return True
    if not hashed_password.startswith("$argon2"):
        return True
    if ARGON2_AVAILABLE and _argon2_hasher:
        try:
            return _argon2_hasher.check_needs_rehash(hashed_password)
        except Exception:
            return True
    return False


def generate_session_id() -> str:
    """Generate a high-entropy, URL-safe random session identifier."""
    return secrets.token_urlsafe(32)


def hash_session_id(session_id: str) -> str:
    """Compute SHA-256 hash of a session identifier for database storage.

    Storing hashed session tokens in PostgreSQL ensures that an unauthorized database
    dump does not expose active user session credentials.
    """
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def generate_csrf_token() -> str:
    """Generate a cryptographic CSRF token for double-submit / header verification."""
    return secrets.token_hex(32)


def verify_csrf_token(header_token: Optional[str], cookie_token: Optional[str]) -> bool:
    """Verify that the submitted CSRF header matches the CSRF cookie securely."""
    if not header_token or not cookie_token:
        return False
    return secrets.compare_digest(header_token.strip(), cookie_token.strip())
