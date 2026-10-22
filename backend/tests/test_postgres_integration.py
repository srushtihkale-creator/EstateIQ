"""PostgreSQL Integration Tests for EstateIQ Backend.

Tests live PostgreSQL connectivity, Alembic schema tables, Argon2id hashing,
server-managed sessions, user data isolation, prediction history persistence,
health/readiness probes, and ML model inference against the actual local PostgreSQL database.
"""

import json
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.db.session import SessionLocal, engine
from backend.app.models import User, UserSession, PredictionHistory

is_postgres = settings.DATABASE_URL.startswith("postgresql")

pytestmark = pytest.mark.skipif(
    not is_postgres,
    reason="PostgreSQL not configured in DATABASE_URL. Skipping live PostgreSQL integration tests.",
)


@pytest.fixture
def pg_client():
    """TestClient that uses live PostgreSQL without SQLite dependency overrides."""
    app.dependency_overrides.clear()
    with TestClient(app) as client:
        yield client


@pytest.fixture
def property_payload():
    """Standard valid property prediction payload."""
    return {
        "POSTED_BY": "Owner",
        "UNDER_CONSTRUCTION": 0,
        "RERA": 1,
        "BHK_NO.": 2,
        "BHK_OR_RK": "BHK",
        "SQUARE_FT": 1250.0,
        "READY_TO_MOVE": 1,
        "RESALE": 1,
        "ADDRESS": "Whitefield, Bangalore",
        "LONGITUDE": 77.59796,
        "LATITUDE": 12.96991,
    }


def test_postgres_connectivity_and_schema():
    """Verify live PostgreSQL database connectivity and Alembic schema tables."""
    with engine.connect() as conn:
        dialect = conn.dialect.name
        assert dialect == "postgresql"
        db_name = conn.execute(text("SELECT current_database()")).scalar()
        assert db_name == "estateiq_db"

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    for expected in ["alembic_version", "users", "user_sessions", "prediction_history"]:
        assert expected in tables

    with engine.connect() as conn:
        rev = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
        assert rev == "001_initial_schema"


def test_postgres_health_and_readiness_endpoints(pg_client):
    """Verify /health and /ready endpoints against live PostgreSQL."""
    res_h = pg_client.get("/health")
    assert res_h.status_code == 200
    assert res_h.json()["database"] == "connected"

    res_r = pg_client.get("/ready")
    assert res_r.status_code == 200
    data_r = res_r.json()
    assert data_r["status"] == "ready"
    assert data_r["database"] == "connected"
    assert data_r["model_artifact"] == "loaded"


def test_postgres_model_metrics_endpoint(pg_client):
    """Verify /api/model/metrics returns verified evaluation metadata."""
    res = pg_client.get("/api/model/metrics")
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "DecisionTree_Raw"
    assert data["val_rmse_lacs"] > 0
    assert data["val_r2_score"] > 0.80


def test_postgres_auth_argon2id_and_session_lifecycle(pg_client):
    """Verify registration, Argon2id hashing, login, profile, and logout in PostgreSQL."""
    test_email = f"test_pg_lifecycle_{uuid.uuid4().hex[:8]}@example.com"
    test_password = "SecurePassword123!"
    user_id = None

    try:
        # 1. Register
        res_reg = pg_client.post(
            "/api/auth/register",
            json={
                "email": test_email,
                "password": test_password,
                "full_name": "PG Lifecycle User",
            },
        )
        assert res_reg.status_code == 201
        reg_data = res_reg.json()
        user_id = reg_data["user"]["id"]
        csrf_token = reg_data["csrf_token"]
        cookies = res_reg.cookies
        assert settings.SESSION_COOKIE_NAME in cookies

        # 2. Verify Argon2id hash in PostgreSQL users table
        db = SessionLocal()
        try:
            db_user = db.query(User).filter(User.id == user_id).first()
            assert db_user is not None
            assert db_user.hashed_password.startswith("$argon2id$")
            assert test_password not in db_user.hashed_password

            # Verify session in user_sessions table
            sessions = db.query(UserSession).filter(UserSession.user_id == user_id).all()
            assert len(sessions) == 1
            assert len(sessions[0].session_id_hash) == 64
            assert not sessions[0].revoked
        finally:
            db.close()

        # 3. Authenticate with custom client carrying session cookie
        auth_client = TestClient(app, cookies=cookies)
        res_me = auth_client.get("/api/auth/me")
        assert res_me.status_code == 200
        assert res_me.json()["id"] == user_id

        # 4. Logout
        res_logout = auth_client.post(
            "/api/auth/logout",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert res_logout.status_code == 200

        # Verify session is marked revoked in PostgreSQL
        db = SessionLocal()
        try:
            active = db.query(UserSession).filter(
                UserSession.user_id == user_id,
                UserSession.revoked == False,
            ).all()
            assert len(active) == 0

            revoked = db.query(UserSession).filter(
                UserSession.user_id == user_id,
                UserSession.revoked == True,
            ).all()
            assert len(revoked) == 1
        finally:
            db.close()

        # Subsequent /me fails
        res_me_after = auth_client.get("/api/auth/me")
        assert res_me_after.status_code == 401

    finally:
        # Cleanup
        if user_id:
            db = SessionLocal()
            try:
                db.query(UserSession).filter(UserSession.user_id == user_id).delete()
                db.query(User).filter(User.id == user_id).delete()
                db.commit()
            except Exception:
                db.rollback()
            finally:
                db.close()


def test_postgres_prediction_persistence_and_isolation(pg_client, property_payload):
    """Verify prediction creation, PostgreSQL persistence, and strict multi-user isolation."""
    user_a_email = f"user_a_{uuid.uuid4().hex[:8]}@example.com"
    user_b_email = f"user_b_{uuid.uuid4().hex[:8]}@example.com"
    password = "TestPassword123!"
    created_uids = []

    try:
        # Register User A
        res_a = pg_client.post(
            "/api/auth/register",
            json={"email": user_a_email, "password": password, "full_name": "User Alpha"},
        )
        assert res_a.status_code == 201
        user_a_id = res_a.json()["user"]["id"]
        csrf_a = res_a.json()["csrf_token"]
        cookies_a = res_a.cookies
        created_uids.append(user_a_id)

        # Register User B
        res_b = pg_client.post(
            "/api/auth/register",
            json={"email": user_b_email, "password": password, "full_name": "User Beta"},
        )
        assert res_b.status_code == 201
        user_b_id = res_b.json()["user"]["id"]
        cookies_b = res_b.cookies
        created_uids.append(user_b_id)

        client_a = TestClient(app, cookies=cookies_a)
        client_b = TestClient(app, cookies=cookies_b)

        # User A makes prediction
        res_pred = client_a.post(
            "/api/predictions",
            json=property_payload,
            headers={"X-CSRF-Token": csrf_a},
        )
        assert res_pred.status_code == 200
        pred_data = res_pred.json()
        assert pred_data["estimated_price_lacs"] > 0
        pred_id = pred_data["prediction_id"]
        assert pred_id is not None

        # Verify record in PostgreSQL prediction_history
        db = SessionLocal()
        try:
            records = db.query(PredictionHistory).filter(PredictionHistory.user_id == user_a_id).all()
            assert len(records) == 1
            assert records[0].id == pred_id
            assert records[0].predicted_price_lacs == pred_data["estimated_price_lacs"]
            inputs = json.loads(records[0].inputs)
            assert inputs["BHK_NO."] == 2
        finally:
            db.close()

        # User A views history -> has 1 record
        res_hist_a = client_a.get("/api/predictions")
        assert res_hist_a.status_code == 200
        assert res_hist_a.json()["total"] == 1
        assert res_hist_a.json()["items"][0]["id"] == pred_id

        # User B views history -> 0 records (isolation)
        res_hist_b = client_b.get("/api/predictions")
        assert res_hist_b.status_code == 200
        assert res_hist_b.json()["total"] == 0

        # User B attempts to access User A's prediction by ID -> 404
        res_cross = client_b.get(f"/api/predictions/{pred_id}")
        assert res_cross.status_code == 404

    finally:
        # Cleanup
        db = SessionLocal()
        try:
            for uid in created_uids:
                db.query(PredictionHistory).filter(PredictionHistory.user_id == uid).delete()
                db.query(UserSession).filter(UserSession.user_id == uid).delete()
                db.query(User).filter(User.id == uid).delete()
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()
