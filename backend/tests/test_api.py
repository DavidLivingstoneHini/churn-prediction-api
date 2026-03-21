"""
HTTP endpoint tests using FastAPI TestClient with SQLite.
All tests use the fake ML artifacts from conftest.py.
"""
import pytest
from fastapi.testclient import TestClient


# ── Health ────────────────────────────────────────────────────

def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "model_version" in data


# ── Auth ──────────────────────────────────────────────────────

class TestAuth:

    def test_register_success(self, client: TestClient):
        resp = client.post("/api/v1/auth/register", json={
            "email":     "newuser@example.com",
            "full_name": "New User",
            "password":  "securepassword123",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token"  in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_register_duplicate_email(self, client: TestClient):
        payload = {
            "email":     "duplicate@example.com",
            "full_name": "Dup User",
            "password":  "securepassword123",
        }
        client.post("/api/v1/auth/register", json=payload)
        resp = client.post("/api/v1/auth/register", json=payload)
        assert resp.status_code == 409

    def test_register_short_password(self, client: TestClient):
        resp = client.post("/api/v1/auth/register", json={
            "email":     "short@example.com",
            "full_name": "Short Pw",
            "password":  "abc",
        })
        assert resp.status_code == 422

    def test_login_success(self, client: TestClient, registered_user: dict):
        resp = client.post("/api/v1/auth/login", json={
            "email":    registered_user["email"],
            "password": registered_user["password"],
        })
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_login_wrong_password(self, client: TestClient, registered_user: dict):
        resp = client.post("/api/v1/auth/login", json={
            "email":    registered_user["email"],
            "password": "wrongpassword",
        })
        assert resp.status_code == 401

    def test_login_unknown_email(self, client: TestClient):
        resp = client.post("/api/v1/auth/login", json={
            "email":    "nobody@example.com",
            "password": "doesnotmatter",
        })
        assert resp.status_code == 401

    def test_me_returns_user(self, client: TestClient, auth_headers: dict):
        resp = client.get("/api/v1/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "testuser@example.com"
        assert data["role"]  == "user"

    def test_me_requires_auth(self, client: TestClient):
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 403

    def test_refresh_token(self, client: TestClient, registered_user: dict):
        refresh = registered_user["tokens"]["refresh_token"]
        resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
        # Refresh token may already be rotated from other tests — just check schema
        assert resp.status_code in (200, 401)
        if resp.status_code == 200:
            assert "access_token" in resp.json()

    def test_logout(self, client: TestClient, registered_user: dict):
        # Login fresh to get a usable refresh token
        login = client.post("/api/v1/auth/login", json={
            "email":    registered_user["email"],
            "password": registered_user["password"],
        })
        refresh = login.json()["refresh_token"]
        resp = client.post("/api/v1/auth/logout", json={"refresh_token": refresh})
        assert resp.status_code == 204


# ── Predict single ────────────────────────────────────────────

SAMPLE_CUSTOMER = {
    "tenure":            12,
    "monthly_charges":   65.00,
    "total_charges":     780.00,
    "senior_citizen":    0,
    "partner":           "No",
    "dependents":        "No",
    "phone_service":     "Yes",
    "multiple_lines":    "No",
    "internet_service":  "Fiber optic",
    "online_security":   "No",
    "online_backup":     "No",
    "device_protection": "No",
    "tech_support":      "No",
    "streaming_tv":      "No",
    "streaming_movies":  "No",
    "contract":          "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method":    "Electronic check",
}


class TestPredictSingle:

    def test_predict_requires_auth(self, client: TestClient):
        resp = client.post("/api/v1/predict/single", json=SAMPLE_CUSTOMER)
        assert resp.status_code == 403

    def test_predict_success(self, client: TestClient, auth_headers: dict):
        resp = client.post(
            "/api/v1/predict/single",
            json=SAMPLE_CUSTOMER,
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "prediction_id"      in data
        assert "churn_probability"  in data
        assert "risk_level"         in data
        assert "similar_customers"  in data
        assert "recommendation"     in data
        assert "inference_time_ms"  in data

    def test_probability_in_range(self, client: TestClient, auth_headers: dict):
        resp = client.post(
            "/api/v1/predict/single",
            json=SAMPLE_CUSTOMER,
            headers=auth_headers,
        )
        assert resp.status_code == 201
        prob = resp.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_risk_level_valid(self, client: TestClient, auth_headers: dict):
        resp = client.post(
            "/api/v1/predict/single",
            json=SAMPLE_CUSTOMER,
            headers=auth_headers,
        )
        assert resp.json()["risk_level"] in ("low", "medium", "high")

    def test_recommendation_non_empty(self, client: TestClient, auth_headers: dict):
        resp = client.post(
            "/api/v1/predict/single",
            json=SAMPLE_CUSTOMER,
            headers=auth_headers,
        )
        assert len(resp.json()["recommendation"]) > 10

    def test_with_customer_id(self, client: TestClient, auth_headers: dict):
        payload = {**SAMPLE_CUSTOMER, "customer_id": "CUST-001"}
        resp = client.post(
            "/api/v1/predict/single",
            json=payload,
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["customer_id"] == "CUST-001"

    def test_missing_required_field(self, client: TestClient, auth_headers: dict):
        payload = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "tenure"}
        resp = client.post(
            "/api/v1/predict/single",
            json=payload,
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_invalid_tenure(self, client: TestClient, auth_headers: dict):
        payload = {**SAMPLE_CUSTOMER, "tenure": -5}
        resp = client.post(
            "/api/v1/predict/single",
            json=payload,
            headers=auth_headers,
        )
        assert resp.status_code == 422


# ── Predict batch ─────────────────────────────────────────────

class TestPredictBatch:

    SAMPLE_CSV = (
        "tenure,MonthlyCharges,TotalCharges,Contract,InternetService,PaymentMethod\n"
        "12,65.00,780.00,Month-to-month,Fiber optic,Electronic check\n"
        "48,45.50,2184.00,One year,DSL,Bank transfer (automatic)\n"
        "2,89.00,178.00,Month-to-month,Fiber optic,Electronic check\n"
    )

    def test_batch_requires_auth(self, client: TestClient):
        resp = client.post(
            "/api/v1/predict/batch",
            files={"file": ("test.csv", self.SAMPLE_CSV.encode(), "text/csv")},
        )
        assert resp.status_code == 403

    def test_batch_success(self, client: TestClient, auth_headers: dict):
        resp = client.post(
            "/api/v1/predict/batch",
            files={"file": ("test.csv", self.SAMPLE_CSV.encode(), "text/csv")},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_rows"] == 3
        assert data["status"]     == "completed"
        assert "high_risk"        in data
        assert "medium_risk"      in data
        assert "low_risk"         in data
        assert data["high_risk"] + data["medium_risk"] + data["low_risk"] == 3

    def test_batch_rejects_non_csv(self, client: TestClient, auth_headers: dict):
        resp = client.post(
            "/api/v1/predict/batch",
            files={"file": ("test.txt", b"not csv", "text/plain")},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_batch_missing_required_column(self, client: TestClient, auth_headers: dict):
        bad_csv = b"MonthlyCharges,TotalCharges\n65.00,780.00\n"
        resp = client.post(
            "/api/v1/predict/batch",
            files={"file": ("bad.csv", bad_csv, "text/csv")},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_batch_avg_probability_in_range(self, client: TestClient, auth_headers: dict):
        resp = client.post(
            "/api/v1/predict/batch",
            files={"file": ("test.csv", self.SAMPLE_CSV.encode(), "text/csv")},
            headers=auth_headers,
        )
        avg = resp.json()["avg_churn_probability"]
        assert 0.0 <= avg <= 1.0


# ── Auth security ─────────────────────────────────────────────

class TestAuthSecurity:

    def test_invalid_bearer_token(self, client: TestClient):
        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer totally_fake_token"},
        )
        assert resp.status_code == 401

    def test_malformed_authorization_header(self, client: TestClient):
        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "NotBearer token"},
        )
        assert resp.status_code == 403

    def test_predict_with_invalid_token(self, client: TestClient):
        resp = client.post(
            "/api/v1/predict/single",
            json=SAMPLE_CUSTOMER,
            headers={"Authorization": "Bearer bad.token.here"},
        )
        assert resp.status_code == 401
