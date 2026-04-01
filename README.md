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

* Docker Desktop (running)
* No other local dependencies — everything runs in containers

## Setup

### 1\. Clone and configure

```bash
git clone https://github.com/DavidLivingstoneHini/churn-prediction-api.git
cd churn-prediction-api
cp env.example .env
```

Edit `.env` — minimum values to fill in:

```env
POSTGRES\\\_PASSWORD=choose\\\_any\\\_strong\\\_password
DATABASE\\\_URL=postgresql+asyncpg://churn\\\_user:choose\\\_any\\\_strong\\\_password@postgres:5432/churn\\\_db

REDIS\\\_PASSWORD=choose\\\_any\\\_redis\\\_password
REDIS\\\_URL=redis://:choose\\\_any\\\_redis\\\_password@redis:6379/0

JWT\\\_SECRET=<run: openssl rand -hex 32>
JWT\\\_REFRESH\\\_SECRET=<run: openssl rand -hex 32>
```

**On Windows PowerShell** (if openssl not available):

```powershell
-join ((1..32) | ForEach-Object { '{0:x2}' -f (Get-Random -Max 256) })
```

Run twice — once for each secret.

### 2\. Start the application

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
\\\[1/6] Generating training dataset... 7043 rows | churn rate: 26.5%
\\\[4/6] Training XGBoost + calibration...
\\\[5/6] Evaluating model...
      AUC-ROC:   0.91xx
      Precision: 0.87xx
\\\[6/6] Building FAISS index... Index contains 5634 vectors
✅ Training complete.
```

Once running:

* **Frontend:** http://localhost:3000
* **API docs:** http://localhost:8000/docs
* **Health:** http://localhost:8000/health

### 3\. Make yourself admin (run once)

```bash
docker exec -it churn\\\_postgres psql -U churn\\\_user -d churn\\\_db \\\\
  -c "UPDATE users SET role='admin' WHERE email='your@email.com';"
```

Log out and back in to pick up the admin role.

## How the ML Pipeline Works

### Feature Engineering

Raw Telco customer data is transformed into 22 model-ready features:

|Category|Features|
|-|-|
|Numeric|tenure, monthly\_charges, total\_charges|
|Engineered|num\_services, avg\_monthly\_spend, charge\_per\_service, tenure\_monthly\_ratio|
|Binary|senior\_citizen, partner, dependents, phone\_service, multiple\_lines, online\_security, online\_backup, device\_protection, tech\_support, streaming\_tv, streaming\_movies, paperless\_billing|
|One-hot|internet\_service (×3), contract (×3), payment\_method (×4)|

Engineered features capture churn-relevant behaviours:

* `charge\\\_per\\\_service` — high cost relative to services used signals dissatisfaction
* `tenure\\\_monthly\\\_ratio` — stable long-term customers pay proportionally less over time
* `avg\\\_monthly\\\_spend` — catches mid-contract price increases

### Model

* **Algorithm:** XGBoost with `scale\\\_pos\\\_weight` for class imbalance (73/27 split)
* **Calibration:** Platt scaling via `CalibratedClassifierCV` for reliable probabilities
* **Evaluation:** Stratified 80/20 split, AUC-ROC as primary metric

### FAISS Similarity Search

Every prediction returns the top-5 most similar training customers using FAISS IVF (Inverted File Index). Provides CRM teams with real historical precedent for each prediction rather than a bare probability.

### PSI Drift Detection

Population Stability Index compares training vs. recent inference distributions:

|PSI|Status|
|-|-|
|< 0.1|Stable|
|0.1 – 0.2|Monitor|
|≥ 0.2|Retrain|

## API Reference

### Auth

|Method|Endpoint|Description|
|-|-|-|
|POST|`/api/v1/auth/register`|Register|
|POST|`/api/v1/auth/login`|Login|
|POST|`/api/v1/auth/refresh`|Refresh token|
|POST|`/api/v1/auth/logout`|Revoke token|
|GET|`/api/v1/auth/me`|Current user|

### Predictions

|Method|Endpoint|Description|
|-|-|-|
|POST|`/api/v1/predict/single`|Single customer prediction|
|POST|`/api/v1/predict/batch`|Batch CSV prediction|

### Admin

|Method|Endpoint|Description|
|-|-|-|
|GET|`/api/v1/admin/model`|Model metrics|
|GET|`/api/v1/admin/analytics`|Dashboard data|
|GET|`/api/v1/admin/drift`|PSI drift report|
|GET|`/api/v1/admin/predictions/history`|Audit log|

Full interactive docs at http://localhost:8000/docs

## Sample Request

```bash
curl -X POST http://localhost:8001/api/v1/predict/single \\\\
  -H "Authorization: Bearer <token>" \\\\
  -H "Content-Type: application/json" \\\\
  -d '{
    "tenure": 6,
    "monthly\\\_charges": 89.00,
    "total\\\_charges": 534.00,
    "contract": "Month-to-month",
    "internet\\\_service": "Fiber optic",
    "payment\\\_method": "Electronic check",
    "paperless\\\_billing": "Yes",
    "senior\\\_citizen": 0,
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
tests/test\\\_features.py::test\\\_basic\\\_feature\\\_engineering PASSED
tests/test\\\_features.py::test\\\_engineered\\\_features PASSED
tests/test\\\_features.py::test\\\_feature\\\_count PASSED
tests/test\\\_features.py::test\\\_single\\\_inference PASSED
tests/test\\\_drift.py::test\\\_psi\\\_identical\\\_distributions PASSED
tests/test\\\_drift.py::test\\\_psi\\\_different\\\_distributions PASSED
tests/test\\\_drift.py::test\\\_drift\\\_detected PASSED
tests/test\\\_drift.py::test\\\_no\\\_drift PASSED
tests/test\\\_predictor.py::test\\\_predictor\\\_loads PASSED
tests/test\\\_predictor.py::test\\\_prediction\\\_output PASSED
tests/test\\\_predictor.py::test\\\_risk\\\_levels PASSED
tests/test\\\_predictor.py::test\\\_similar\\\_customers PASSED
tests/test\\\_api.py::test\\\_health PASSED
tests/test\\\_api.py::test\\\_register PASSED
tests/test\\\_api.py::test\\\_login PASSED
tests/test\\\_api.py::test\\\_predict\\\_single PASSED
tests/test\\\_api.py::test\\\_predict\\\_requires\\\_auth PASSED
```

## Tech Stack

|Layer|Technology|
|-|-|
|Frontend|React 18, TypeScript, Vite, Tailwind CSS, Recharts|
|Backend|FastAPI, SQLAlchemy (async)|
|ML|XGBoost, scikit-learn, FAISS, Pandas, NumPy|
|Drift|PSI via SciPy|
|Database|PostgreSQL 16|
|Cache|Redis 7|
|Auth|JWT (HS256), bcrypt|
|Infra|Docker, Docker Compose, Nginx|



