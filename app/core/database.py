import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool
from app.core.config import get_settings

settings = get_settings()

# 1. Async Database Engine (asyncpg requires '?ssl=require' NOT '?sslmode=require')
raw_async_url = settings.NEON_DB_ASYNC_URL or settings.DATABASE_URL or ""
if raw_async_url.startswith("postgres://"):
    raw_async_url = raw_async_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif raw_async_url.startswith("postgresql://") and not raw_async_url.startswith("postgresql+asyncpg://"):
    raw_async_url = raw_async_url.replace("postgresql://", "postgresql+asyncpg://", 1)

# Clean asyncpg query parameters: replace sslmode with ssl
if "sslmode=" in raw_async_url:
    raw_async_url = raw_async_url.replace("sslmode=", "ssl=")

async_engine = create_async_engine(
    raw_async_url,
    poolclass=NullPool,
    echo=False
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# 2. Synchronous Database Engine (psycopg2 requires '?sslmode=require')
raw_sync_url = settings.NEON_DB_SYNC_URL or settings.SYNC_DATABASE_URL or settings.DATABASE_URL or ""
if raw_sync_url.startswith("postgresql+asyncpg://"):
    raw_sync_url = raw_sync_url.replace("postgresql+asyncpg://", "postgresql://", 1)
elif raw_sync_url.startswith("postgres://"):
    raw_sync_url = raw_sync_url.replace("postgres://", "postgresql://", 1)

if "?ssl=require" in raw_sync_url:
    raw_sync_url = raw_sync_url.replace("?ssl=require", "?sslmode=require")
elif "&ssl=require" in raw_sync_url:
    raw_sync_url = raw_sync_url.replace("&ssl=require", "&sslmode=require")

sync_engine = create_engine(
    raw_sync_url,
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
