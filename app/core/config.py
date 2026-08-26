from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

# Resolve the absolute path to the root directory: financial-rag-system/.env
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    """
    Application Settings loaded dynamically from .env.
    Uses Pydantic BaseSettings for strict type validation.
    """
    APP_ENV: str = "development"
    DEBUG: bool = True
    PROJECT_NAME: str = "Financial RAG & Analytics Platform"

    # Database connection strings
    DATABASE_URL: str
    SYNC_DATABASE_URL: str

    # Google AI Studio / Gemini API Configurations
    GEMINI_API_KEY: str
    GEMINI_GENERATIVE_MODEL: str = "gemini-1.5-pro"
    GEMINI_FAST_MODEL: str = "gemini-1.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"
    EMBEDDING_DIMENSION: int = 768

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache()
def get_settings() -> Settings:
    """Returns a cached instance of application settings."""
    return Settings()