import os
import hashlib
from typing import Optional, Dict, Tuple

DEFAULT_USERS = {
    "admin": {
        "password_hash": hashlib.sha256(os.getenv("ADMIN_PASSWORD", "Admin2026!").encode("utf-8")).hexdigest(),
        "role": "admin",
        "name": "Portfolio Administrator"
    },
    "analyst": {
        "password_hash": hashlib.sha256(os.getenv("ANALYST_PASSWORD", "Analyst2026!").encode("utf-8")).hexdigest(),
        "role": "analyst",
        "name": "Research Analyst"
    }
}

ADMIN_INGESTION_KEY = os.getenv("ADMIN_INGESTION_KEY", "IngestMaster2026!")

def verify_credentials(username: str, password: str) -> Optional[Tuple[str, str, str]]:
    """
    Verifies user credentials and returns (username, role, display_name) if valid.
    """
    u = username.strip().lower()
    p_hash = hashlib.sha256(password.strip().encode("utf-8")).hexdigest()

    if u in DEFAULT_USERS and DEFAULT_USERS[u]["password_hash"] == p_hash:
        return u, DEFAULT_USERS[u]["role"], DEFAULT_USERS[u]["name"]
    return None

def verify_ingestion_passkey(passkey: str) -> bool:
    """Verifies standalone ingestion passkey."""
    return passkey.strip() == ADMIN_INGESTION_KEY.strip()
