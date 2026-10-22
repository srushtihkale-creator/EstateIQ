"""Integration tests for Authentication, Session Management, and User Data Isolation."""

import pytest
from fastapi import status
from backend.app.core.config import settings


def test_user_registration_success(client):
    """Verify registration creates user and sets secure HttpOnly session and CSRF cookies."""
    payload = {
        "email": "newuser@example.com",
        "password": "StrongPassword123!",
        "full_name": "New User",
    }
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()
    assert data["user"]["email"] == "newuser@example.com"
    assert "id" in data["user"]
    assert "csrf_token" in data

    # Verify cookies
    cookies = response.cookies
    assert settings.SESSION_COOKIE_NAME in cookies
    assert settings.CSRF_COOKIE_NAME in cookies


def test_user_registration_duplicate_email(client):
    """Verify registration rejects duplicate email addresses with HTTP 400."""
    payload = {
        "email": "duplicate@example.com",
        "password": "Password123!",
    }
    res1 = client.post("/api/auth/register", json=payload)
    assert res1.status_code == status.HTTP_201_CREATED

    res2 = client.post("/api/auth/register", json=payload)
    assert res2.status_code == status.HTTP_400_BAD_REQUEST
    assert "already exists" in res2.json()["error"]["message"]


def test_user_login_success_and_logout(client):
    """Verify login authenticates credentials and logout revokes session and clears cookies."""
    # Register first
    client.post(
        "/api/auth/register",
        json={"email": "login_test@example.com", "password": "Password123!"},
    )

    # Login
    login_res = client.post(
        "/api/auth/login",
        json={"email": "login_test@example.com", "password": "Password123!"},
    )
    assert login_res.status_code == status.HTTP_200_OK
    assert settings.SESSION_COOKIE_NAME in login_res.cookies

    # Profile check
    me_res = client.get("/api/auth/me")
    assert me_res.status_code == status.HTTP_200_OK
    assert me_res.json()["email"] == "login_test@example.com"

    # Logout
    logout_res = client.post("/api/auth/logout")
    assert logout_res.status_code == status.HTTP_200_OK

    # Profile check after logout must fail
    me_after_res = client.get("/api/auth/me")
    assert me_after_res.status_code == status.HTTP_401_UNAUTHORIZED


def test_user_login_invalid_password(client):
    """Verify login rejects invalid password with generic 401 Unauthorized."""
    client.post(
        "/api/auth/register",
        json={"email": "wrongpass@example.com", "password": "CorrectPassword123!"},
    )

    bad_res = client.post(
        "/api/auth/login",
        json={"email": "wrongpass@example.com", "password": "WrongPassword!"},
    )
    assert bad_res.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid email or password" in bad_res.json()["error"]["message"]


def test_strict_user_data_isolation(client, authenticated_user_a, authenticated_user_b):
    """Verify that User A cannot see or access User B's prediction history records."""
    property_payload = {
        "POSTED_BY": "Owner",
        "UNDER_CONSTRUCTION": 0,
        "RERA": 1,
        "BHK_NO.": 2,
        "BHK_OR_RK": "BHK",
        "SQUARE_FT": 1200.0,
        "RESALE": 1,
        "ADDRESS": "Whitefield, Bangalore",
    }

    # 1. User A makes a prediction
    client.cookies.set(settings.SESSION_COOKIE_NAME, authenticated_user_a["session_id"])
    res_a = client.post("/api/predictions", json=property_payload)
    assert res_a.status_code == status.HTTP_200_OK
    pred_id_a = res_a.json()["prediction_id"]
    assert pred_id_a is not None

    # 2. User B makes a prediction
    client.cookies.set(settings.SESSION_COOKIE_NAME, authenticated_user_b["session_id"])
    res_b = client.post("/api/predictions", json=property_payload)
    assert res_b.status_code == status.HTTP_200_OK
    pred_id_b = res_b.json()["prediction_id"]
    assert pred_id_b is not None

    # 3. User B views their history -> must only see pred_id_b
    history_b = client.get("/api/predictions").json()
    b_ids = [item["id"] for item in history_b["items"]]
    assert pred_id_b in b_ids
    assert pred_id_a not in b_ids

    # 4. User B attempts to access User A's prediction record directly -> must return 404
    cross_access_res = client.get(f"/api/predictions/{pred_id_a}")
    assert cross_access_res.status_code == status.HTTP_404_NOT_FOUND

    # 5. User A views their history -> must only see pred_id_a
    client.cookies.set(settings.SESSION_COOKIE_NAME, authenticated_user_a["session_id"])
    history_a = client.get("/api/predictions").json()
    a_ids = [item["id"] for item in history_a["items"]]
    assert pred_id_a in a_ids
    assert pred_id_b not in a_ids

    # 6. User A attempts to access User B's prediction record directly -> must return 404
    cross_access_res2 = client.get(f"/api/predictions/{pred_id_b}")
    assert cross_access_res2.status_code == status.HTTP_404_NOT_FOUND


def test_argon2id_enforced_for_new_registrations(client, db_session):
    """Verify that all new user registrations strictly use Argon2id password hashing."""
    from backend.app.models.user import User
    from backend.app.core.security import verify_password, hash_password

    email = "argon2_test@example.com"
    raw_pass = "Argon2SecurePassword123!"

    reg_res = client.post(
        "/api/auth/register",
        json={"email": email, "password": raw_pass, "full_name": "Argon2 User"},
    )
    assert reg_res.status_code == status.HTTP_201_CREATED

    user = db_session.query(User).filter(User.email == email).first()
    assert user is not None
    # Must strictly be Argon2id ($argon2id$v=19$m=65536,t=3,p=4...)
    assert user.hashed_password.startswith("$argon2id$")
    assert verify_password(raw_pass, user.hashed_password) is True
    assert verify_password("WrongPassword!", user.hashed_password) is False

    # Direct hash_password check
    direct_hash = hash_password("TestPlain123!")
    assert direct_hash.startswith("$argon2id$")


def test_legacy_bcrypt_login_and_auto_rehash(client, db_session):
    """Verify that legacy bcrypt password hashes authenticate and are automatically upgraded to Argon2id."""
    from passlib.context import CryptContext
    from backend.app.models.user import User
    from backend.app.core.security import verify_password

    email = "legacy_bcrypt@example.com"
    plain_pass = "LegacyPassword456!"

    # Seed legacy bcrypt hash directly in DB
    bcrypt_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
    legacy_hash = bcrypt_ctx.hash(plain_pass)
    assert legacy_hash.startswith("$2")

    legacy_user = User(
        email=email,
        hashed_password=legacy_hash,
        full_name="Legacy User",
        role="user",
        is_active=True,
    )
    db_session.add(legacy_user)
    db_session.commit()
    db_session.refresh(legacy_user)

    # Pre-condition: user has legacy bcrypt hash
    assert legacy_user.hashed_password.startswith("$2")

    # Authenticate via API
    login_res = client.post(
        "/api/auth/login",
        json={"email": email, "password": plain_pass},
    )
    assert login_res.status_code == status.HTTP_200_OK

    # Post-condition: hash has been automatically upgraded to Argon2id!
    db_session.refresh(legacy_user)
    assert legacy_user.hashed_password.startswith("$argon2id$")
    assert verify_password(plain_pass, legacy_user.hashed_password) is True

