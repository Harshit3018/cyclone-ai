"""
Application configuration using pydantic-settings.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_ENV: str = "demo"
    APP_PORT: int = 8000
    APP_HOST: str = "0.0.0.0"
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    DATABASE_URL: str = "sqlite:///./cyclone_ai.db"
    MODEL_PATH: str = "models/checkpoints"
    DATA_PATH: str = "data"
    LOG_LEVEL: str = "INFO"

    # External APIs
    IMD_API_KEY: str = ""

    # Frontend serving (production)
    SERVE_FRONTEND: bool = False
    FRONTEND_DIR: str = "frontend/dist"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Use project root .env
_project_root = Path(__file__).parent.parent.parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    settings = Settings(_env_file=str(_env_path))
else:
    settings = Settings()
