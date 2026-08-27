import os
import re
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool
from app.core.config import get_settings

settings = get_settings()

def clean_asyncpg_url(url: str) -> str:
    """
    Cleans database URL specifically for asyncpg.
    asyncpg strictly forbids 'sslmode' query parameter and requires 'ssl=require' or 'postgresql+asyncpg://'.
    """
    if not url:
        return ""
    
    # 1. Ensure driver is postgresql+asyncpg://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    # 2. Replace sslmode= with ssl=
    url = re.sub(r'sslmode=([a-zA-Z0-9_\-]+)', r'ssl=\1', url)
    return url

def clean_psycopg2_url(url: str) -> str:
    """
    Cleans database URL specifically for psycopg2 (synchronous driver).
    psycopg2 requires 'postgresql://' and 'sslmode=require'.
    """
    if not url:
        return ""

    # 1. Ensure driver is postgresql://
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # 2. Ensure sslmode is used instead of ssl
    if "?ssl=require" in url:
        url = url.replace("?ssl=require", "?sslmode=require")
    elif "&ssl=require" in url:
        url = url.replace("&ssl=require", "&sslmode=require")
    elif "sslmode=" not in url and ("neon.tech" in url or "ssl=true" in url.lower()):
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=require"
        
    return url

raw_async = settings.NEON_DB_ASYNC_URL or settings.DATABASE_URL or ""
raw_sync = settings.NEON_DB_SYNC_URL or settings.SYNC_DATABASE_URL or settings.DATABASE_URL or ""

cleaned_async_url = clean_asyncpg_url(raw_async)
cleaned_sync_url = clean_psycopg2_url(raw_sync)

# 1. Asynchronous Database Engine
async_engine = create_async_engine(
    cleaned_async_url,
    poolclass=NullPool,
    echo=False
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# 2. Synchronous Database Engine
sync_engine = create_engine(
    cleaned_sync_url,
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
