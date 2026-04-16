from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/anomalert"
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "anomalert_xgb"
    model_version: str = "v1.2.3"
    secret_key: str = "change-me"
    environment: str = "dev"
    model_local_path: str = "models/model.xgb"
    signed_by_default: str = "system@anomalert-bio"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
