# ML Customer Churn Prediction API

A production-grade machine learning system for real-time customer churn prediction. Built with XGBoost, FastAPI, FAISS similarity search, PSI-based drift detection, and a React analytics dashboard.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│  Single Predict │ Batch CSV Upload │ Analytics │ Drift Monitor  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP / REST
┌────────────────────────────▼────────────────────────────────────┐
│                       FastAPI Backend                           │
│                                                                 │
│  ┌─────────────┐   ┌──────────────────────────────────────────┐ │
│  │  JWT Auth   │   │            ML Pipeline                   │ │
│  │  bcrypt     │   │  Features → XGBoost → Calibration        │ │
│  └─────────────┘   │  FAISS similarity search                  │ │
│                    │  PSI drift detection                       │ │
│                    └──────────────────────────────────────────┘ │
└──────┬──────────────────────────┬───────────────────────────────┘
       │                          │
┌──────▼──────────┐   ┌───────────▼──────────────────────────────┐
│   PostgreSQL    │   │             FAISS Index                   │
│  Users          │   │  IVF flat index — 22 dimensions           │
│  Predictions    │   │  Approx nearest-neighbour search          │
│  Batch jobs     │   └───────────────────────────────────────────┘
│  Drift logs     │
└─────────────────┘
```

## Prerequisites

- Docker Desktop (running)
- No other local dependencies — everything runs in containers

## Setup

### 1. Clone and configure

```bash
git clone https://github.com/DavidLivingstoneHini/churn-prediction-api.git
cd churn-prediction-api
cp env.example .env
```

Edit `.env` — minimum values to fill in:

```env
POSTGRES_PASSWORD=choose_any_strong_password
DATABASE_URL=postgresql+asyncpg://churn_user:choose_any_strong_password@postgres:5432/churn_db

REDIS_PASSWORD=choose_any_redis_password
REDIS_URL=redis://:choose_any_redis_password@redis:6379/0

JWT_SECRET=<run: openssl rand -hex 32>
JWT_REFRESH_SECRET=<run: openssl rand -hex 32>
```

**On Windows PowerShell** (if openssl not available):
```powershell
-join ((1..32) | ForEach-Object { '{0:x2}' -f (Get-Random -Max 256) })
```
Run twice — once for each secret.

### 2. Start the application

```bash
docker-compose up --build
```

The Docker build automatically:
1. Installs all Python dependencies
2. Generates a synthetic Telco Churn dataset (7,043 rows, 26.5% churn rate)
3. Engineers 22 features including derived signals
4. Trains XGBoost with Platt scaling calibration
5. Builds a FAISS IVF index for similarity search
6. Saves all artifacts to a persistent Docker volume

You will see in the build output:
```
[1/6] Generating training dataset... 7043 rows | churn rate: 26.5%
[4/6] Training XGBoost + calibration...
[5/6] Evaluating model...
      AUC-ROC:   0.91xx
      Precision: 0.87xx
[6/6] Building FAISS index... Index contains 5634 vectors
✅ Training complete.
```

Once running:
- **Frontend:** http://localhost:3001
- **API docs:** http://localhost:8001/docs
- **Health:** http://localhost:8001/health

### 3. Make yourself admin (run once)

```bash
docker exec -it churn_postgres psql -U churn_user -d churn_db \
  -c "UPDATE users SET role='admin' WHERE email='your@email.com';"
```

Log out and back in to pick up the admin role.

## How the ML Pipeline Works

### Feature Engineering

Raw Telco customer data is transformed into 22 model-ready features:

| Category | Features |
|---|---|
| Numeric | tenure, monthly_charges, total_charges |
| Engineered | num_services, avg_monthly_spend, charge_per_service, tenure_monthly_ratio |
| Binary | senior_citizen, partner, dependents, phone_service, multiple_lines, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, paperless_billing |
| One-hot | internet_service (×3), contract (×3), payment_method (×4) |

Engineered features capture churn-relevant behaviours:
- `charge_per_service` — high cost relative to services used signals dissatisfaction
- `tenure_monthly_ratio` — stable long-term customers pay proportionally less over time
- `avg_monthly_spend` — catches mid-contract price increases

### Model

- **Algorithm:** XGBoost with `scale_pos_weight` for class imbalance (73/27 split)
- **Calibration:** Platt scaling via `CalibratedClassifierCV` for reliable probabilities
- **Evaluation:** Stratified 80/20 split, AUC-ROC as primary metric

### FAISS Similarity Search

Every prediction returns the top-5 most similar training customers using FAISS IVF (Inverted File Index). Provides CRM teams with real historical precedent for each prediction rather than a bare probability.

### PSI Drift Detection

Population Stability Index compares training vs. recent inference distributions:

| PSI | Status |
|---|---|
| < 0.1 | Stable |
| 0.1 – 0.2 | Monitor |
| ≥ 0.2 | Retrain |

## API Reference

### Auth
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/auth/register` | Register |
| POST | `/api/v1/auth/login` | Login |
| POST | `/api/v1/auth/refresh` | Refresh token |
| POST | `/api/v1/auth/logout` | Revoke token |
| GET  | `/api/v1/auth/me` | Current user |

### Predictions
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/predict/single` | Single customer prediction |
| POST | `/api/v1/predict/batch`  | Batch CSV prediction |

### Admin
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/admin/model` | Model metrics |
| GET | `/api/v1/admin/analytics` | Dashboard data |
| GET | `/api/v1/admin/drift` | PSI drift report |
| GET | `/api/v1/admin/predictions/history` | Audit log |

Full interactive docs at http://localhost:8001/docs

## Sample Request

```bash
curl -X POST http://localhost:8001/api/v1/predict/single \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 6,
    "monthly_charges": 89.00,
    "total_charges": 534.00,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "paperless_billing": "Yes",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No"
  }'
```

## Running Tests

```bash
cd backend
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx

# Train model first (required for predictor tests)
python -m app.ml.train

pytest tests/ -v
```

Expected output:
```
tests/test_features.py::test_basic_feature_engineering PASSED
tests/test_features.py::test_engineered_features PASSED
tests/test_features.py::test_feature_count PASSED
tests/test_features.py::test_single_inference PASSED
tests/test_drift.py::test_psi_identical_distributions PASSED
tests/test_drift.py::test_psi_different_distributions PASSED
tests/test_drift.py::test_drift_detected PASSED
tests/test_drift.py::test_no_drift PASSED
tests/test_predictor.py::test_predictor_loads PASSED
tests/test_predictor.py::test_prediction_output PASSED
tests/test_predictor.py::test_risk_levels PASSED
tests/test_predictor.py::test_similar_customers PASSED
tests/test_api.py::test_health PASSED
tests/test_api.py::test_register PASSED
tests/test_api.py::test_login PASSED
tests/test_api.py::test_predict_single PASSED
tests/test_api.py::test_predict_requires_auth PASSED
```

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, Recharts |
| Backend | FastAPI, SQLAlchemy (async) |
| ML | XGBoost, scikit-learn, FAISS, Pandas, NumPy |
| Drift | PSI via SciPy |
| Database | PostgreSQL 16 |
| Cache | Redis 7 |
| Auth | JWT (HS256), bcrypt |
| Infra | Docker, Docker Compose, Nginx |
