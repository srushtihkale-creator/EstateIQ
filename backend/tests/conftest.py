"""Pytest fixtures and configuration for EstateIQ Backend tests."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.db.session import Base, get_db
from backend.app.services.auth_service import AuthService
from backend.app.models.user import User

# In-memory SQLite engine for fast, isolated unit/integration tests
TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


from backend.app.core.rate_limit import auth_limiter, prediction_limiter


@pytest.fixture(autouse=True)
def reset_rate_limits():
    """Reset rate limiter state before each test to prevent test runner throttling."""
    auth_limiter.reset()
    prediction_limiter.reset()
    yield
    auth_limiter.reset()
    prediction_limiter.reset()


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create all schema tables before test run and teardown after."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session():
    """Yield an isolated database session per test function with automatic rollback."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session):
    """TestClient with get_db dependency overridden to the test database session."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def authenticated_user_a(db_session):
    """Fixture providing User A with an active session ID and CSRF token."""
    user, session_id, csrf_token = AuthService.register_user(
        db=db_session,
        email="user_a@example.com",
        password="Password123!",
        full_name="User Alpha",
    )
    return {
        "user": user,
        "session_id": session_id,
        "csrf_token": csrf_token,
    }


@pytest.fixture
def authenticated_user_b(db_session):
    """Fixture providing User B with an active session ID and CSRF token."""
    user, session_id, csrf_token = AuthService.register_user(
        db=db_session,
        email="user_b@example.com",
        password="Password456!",
        full_name="User Beta",
    )
    return {
        "user": user,
        "session_id": session_id,
        "csrf_token": csrf_token,
    }
