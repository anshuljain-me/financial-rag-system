import os
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery
from app.core.config import get_settings

settings = get_settings()

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

VALID_API_KEY = os.getenv("API_SECRET_KEY", "financial-rag-prod-secret-key-2026")

async def verify_api_key(
    key_from_header: str = Security(api_key_header),
    key_from_query: str = Security(api_key_query)
) -> str:
    """
    Validates that incoming requests supply a valid institutional API Key.
    """
    key = key_from_header or key_from_query
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Pass 'X-API-Key' header or '?api_key=' parameter."
        )
    if key.strip() != VALID_API_KEY.strip():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or revoked API Key."
        )
    return key
