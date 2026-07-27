"""
CalRetail — FastAPI Application Settings
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from pydantic_settings import BaseSettings


_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    APP_NAME: str = "CalRetail Retail AI Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # The SQLite database is the single source of truth for all datasets.
    # Build it with:  python -m notebooks.build_db
    # Override with CALRETAIL_DB to point the app at a different build (e.g. a
    # full-scale one) without touching the committed demo database.
    DATABASE_PATH: str = os.environ.get(
        "CALRETAIL_DB", str(_ROOT / "data" / "calretail.db"))

    DATA_MODELS_DIR: str = str(_ROOT / "data" / "models")
    OPENAI_API_KEY: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
