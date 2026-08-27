import os
import urllib.parse
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool
from app.core.config import get_settings

settings = get_settings()

def clean_asyncpg_url(raw_url: str) -> str:
    """Sanitizes database URL specifically for asyncpg driver."""
    if not raw_url:
        return ""
    if raw_url.startswith("postgres://"):
        raw_url = "postgresql+asyncpg://" + raw_url[len("postgres://"):]
    elif raw_url.startswith("postgresql://") and not raw_url.startswith("postgresql+asyncpg://"):
        raw_url = "postgresql+asyncpg://" + raw_url[len("postgresql://"):]

    parsed = urllib.parse.urlparse(raw_url)
    query_params = urllib.parse.parse_qs(parsed.query)

    # asyncpg only supports 'ssl', never 'sslmode', 'channel_binding', or 'gssencmode'
    cleaned_params = {}
    for k, v in query_params.items():
        if k == "sslmode":
            val = v[0]
            cleaned_params["ssl"] = ["require" if val in ["require", "verify-ca", "verify-full"] else val]
        elif k == "ssl":
            cleaned_params["ssl"] = v
        elif k in ["channel_binding", "gssencmode"]:
            continue
        else:
            cleaned_params[k] = v

    if "ssl" not in cleaned_params and ("neon.tech" in parsed.netloc or "amazonaws.com" in parsed.netloc):
        cleaned_params["ssl"] = ["require"]

    new_query = urllib.parse.urlencode(cleaned_params, doseq=True)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))

def clean_psycopg2_url(raw_url: str) -> str:
    """Sanitizes database URL specifically for psycopg2 driver."""
    if not raw_url:
        return ""
    if raw_url.startswith("postgresql+asyncpg://"):
        raw_url = "postgresql://" + raw_url[len("postgresql+asyncpg://"):]
    elif raw_url.startswith("postgres://"):
        raw_url = "postgresql://" + raw_url[len("postgres://"):]

    parsed = urllib.parse.urlparse(raw_url)
    query_params = urllib.parse.parse_qs(parsed.query)

    cleaned_params = {}
    for k, v in query_params.items():
        if k == "ssl":
            val = v[0]
            cleaned_params["sslmode"] = ["require" if val in ["require", "true", "1"] else val]
        elif k == "sslmode":
            cleaned_params["sslmode"] = v
        elif k in ["channel_binding", "gssencmode"]:
            continue
        else:
            cleaned_params[k] = v

    if "sslmode" not in cleaned_params and ("neon.tech" in parsed.netloc or "amazonaws.com" in parsed.netloc):
        cleaned_params["sslmode"] = ["require"]

    new_query = urllib.parse.urlencode(cleaned_params, doseq=True)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))

# 1. Async Database Engine (NullPool for thread-safe asynchronous operations)
async_db_url = clean_asyncpg_url(settings.NEON_DB_ASYNC_URL or settings.DATABASE_URL)
async_engine = create_async_engine(
    async_db_url,
    poolclass=NullPool,
    echo=False
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# 2. Synchronous Database Engine (for Streamlit thread-safe reads)
sync_db_url = clean_psycopg2_url(settings.NEON_DB_SYNC_URL or settings.SYNC_DATABASE_URL or settings.DATABASE_URL)
sync_engine = create_engine(
    sync_db_url,
    poolclass=NullPool,
    echo=False
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False
)

class Base(DeclarativeBase):
    pass

async def init_db():
    async with async_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
