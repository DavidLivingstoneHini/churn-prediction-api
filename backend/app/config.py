from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_env: str = "development"
    app_name: str = "ML Customer Churn Prediction API"
    app_version: str = "1.0.0"

    # Database
    database_url: str
    postgres_user: str
    postgres_password: str
    postgres_db: str

    # Redis
    redis_url: str
    redis_password: str

    # JWT
    jwt_secret: str
    jwt_refresh_secret: str
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # ML
    model_path: str = "models/churn_model.pkl"
    scaler_path: str = "models/scaler.pkl"
    faiss_index_path: str = "models/faiss.index"
    training_data_path: str = "models/training_data.csv"
    psi_threshold: float = 0.2
    drift_check_interval_hours: int = 24

    # CORS
    allowed_origins: str = "http://localhost:3001"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
