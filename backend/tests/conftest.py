"""
Shared pytest fixtures.

The test suite is designed to run without Docker — it uses an in-memory
SQLite database and mocks heavy external dependencies (FAISS, trained model)
so all tests pass immediately after `pip install -r requirements.txt`.
"""
import os
import sys

# Ensure backend/ is on the path so `app` is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Set env vars BEFORE any app imports ──────────────────────────────────────
os.environ.setdefault("DATABASE_URL",          "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("REDIS_URL",             "redis://localhost:6379/0")
os.environ.setdefault("REDIS_PASSWORD",        "testpass")
os.environ.setdefault("JWT_SECRET",            "test-jwt-secret-at-least-32-chars-long-ok")
os.environ.setdefault("JWT_REFRESH_SECRET",    "test-refresh-secret-at-least-32-chars-long")
os.environ.setdefault("POSTGRES_USER",         "test")
os.environ.setdefault("POSTGRES_PASSWORD",     "test")
os.environ.setdefault("POSTGRES_DB",           "test")
os.environ.setdefault("OPENAI_API_KEY",        "sk-test")
os.environ.setdefault("PINECONE_API_KEY",      "test-key")
os.environ.setdefault("MODEL_PATH",            "/tmp/churn_model.pkl")
os.environ.setdefault("SCALER_PATH",           "/tmp/scaler.pkl")
os.environ.setdefault("FAISS_INDEX_PATH",      "/tmp/faiss.index")
os.environ.setdefault("TRAINING_DATA_PATH",    "/tmp/training_data.csv")

import asyncio
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# ── Lightweight fake model and scaler ────────────────────────────────────────

class _FakeModel:
    """Mirrors sklearn API — returns fixed probabilities."""
    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, "shape") else 1
        return np.array([[0.3, 0.7]] * n)


class _FakeScaler:
    def transform(self, X):
        return np.array(X, dtype=np.float32)
    def fit_transform(self, X):
        return self.transform(X)


def _write_fake_artifacts():
    """Write lightweight pickle artifacts so the predictor can load them."""
    import json, faiss
    from app.ml.features import ALL_FEATURES

    Path("/tmp").mkdir(exist_ok=True)

    with open("/tmp/churn_model.pkl",  "wb") as f: pickle.dump(_FakeModel(),  f)
    with open("/tmp/scaler.pkl",       "wb") as f: pickle.dump(_FakeScaler(), f)

    # Minimal FAISS index
    dim   = len(ALL_FEATURES)
    index = faiss.IndexFlatL2(dim)
    vecs  = np.random.rand(10, dim).astype(np.float32)
    index.add(vecs)
    faiss.write_index(index, "/tmp/faiss.index")

    np.save("/tmp/train_vectors.npy", vecs)
    np.save("/tmp/train_labels.npy",  np.zeros(10, dtype=np.float32))

    # Tiny training CSV
    rows = []
    for _ in range(50):
        rows.append({
            "tenure": 12, "MonthlyCharges": 65.0, "TotalCharges": 780.0,
            "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "Churn": 0,
        })
    pd.DataFrame(rows).to_csv("/tmp/training_data.csv", index=False)

    meta = {
        "version": "1.0.0", "auc_roc": 0.91, "precision": 0.87,
        "recall": 0.85, "f1_score": 0.86, "training_rows": 40,
        "test_rows": 10, "churn_rate": 0.265,
        "feature_names": ALL_FEATURES,
    }
    Path("/tmp/model_metadata.json").write_text(json.dumps(meta))


_write_fake_artifacts()

# ── In-memory SQLite engine for tests ────────────────────────────────────────

TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_churn.db"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(
    test_engine, class_=AsyncSession,
    expire_on_commit=False, autocommit=False, autoflush=False,
)

# ── Create tables RIGHT NOW at import time ────────────────────────────────────
def _create_tables_sync():
    # Models MUST be imported before create_all so they register with Base.metadata
    import app.database.models  # noqa: F401
    from app.database.session import Base
    async def _run():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    asyncio.run(_run())

_create_tables_sync()


@pytest.fixture(scope="session", autouse=True)
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    """Tables already created at module import. Just clean up after session."""
    yield
    Path("./test_churn.db").unlink(missing_ok=True)


@pytest_asyncio.fixture
async def db_session():
    async with TestSessionLocal() as session:
        yield session
        await session.rollback()


@pytest.fixture(scope="session")
def app(setup_db):
    """FastAPI app with DB dependency overridden to use SQLite."""
    from app.main import app as _app
    from app.database.session import get_db

    async def override_get_db():
        async with TestSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    _app.dependency_overrides[get_db] = override_get_db
    return _app


@pytest.fixture(scope="session")
def client(app):
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture
async def async_client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture(scope="session")
def registered_user(client):
    """Create a test user once per session and return credentials + tokens."""
    resp = client.post("/api/v1/auth/register", json={
        "email":     "testuser@example.com",
        "full_name": "Test User",
        "password":  "testpassword123",
    })
    assert resp.status_code == 201
    return {
        "email":    "testuser@example.com",
        "password": "testpassword123",
        "tokens":   resp.json(),
    }


@pytest.fixture(scope="session")
def auth_headers(registered_user):
    token = registered_user["tokens"]["access_token"]
    return {"Authorization": f"Bearer {token}"}
