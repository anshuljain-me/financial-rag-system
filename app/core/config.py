import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Automatically bridge Streamlit Cloud secrets to environment variables if running in Cloud
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for key in ["NEON_DB_ASYNC_URL", "NEON_DB_SYNC_URL", "GEMINI_API_KEY", "API_SECRET_KEY"]:
            if key in st.secrets and key not in os.environ:
                os.environ[key] = str(st.secrets[key])
except Exception:
    pass

class Settings(BaseSettings):
    PROJECT_NAME: str = "Institutional Financial RAG & SEC Intelligence"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    NEON_DB_ASYNC_URL: str = os.getenv("NEON_DB_ASYNC_URL", "")
    NEON_DB_SYNC_URL: str = os.getenv("NEON_DB_SYNC_URL", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "financial-rag-prod-secret-key-2026")
    EMBEDDING_DIMENSION: int = 768

    class Config:
        env_file = Path(__file__).resolve().parent.parent.parent / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

def get_settings() -> Settings:
    return Settings()
