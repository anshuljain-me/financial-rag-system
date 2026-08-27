import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool
from app.core.config import get_settings

settings = get_settings()

# 1. Async Database Engine (asyncpg requires '?ssl=require' NOT '?sslmode=require')
raw_async = settings.NEON_DB_ASYNC_URL or settings.DATABASE_URL or ""
if raw_async.startswith("postgres://"):
    raw_async = raw_async.replace("postgres://", "postgresql+asyncpg://", 1)
elif raw_async.startswith("postgresql://") and not raw_async.startswith("postgresql+asyncpg://"):
    raw_async = raw_async.replace("postgresql://", "postgresql+asyncpg://", 1)

# Clean query params: replace sslmode with ssl for asyncpg
if "sslmode=" in raw_async:
    raw_async = raw_async.replace("sslmode=", "ssl=")

async_engine = create_async_engine(
    raw_async,
    poolclass=NullPool,
    echo=False
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# 2. Synchronous Database Engine (psycopg2 requires '?sslmode=require')
raw_sync = settings.NEON_DB_SYNC_URL or settings.SYNC_DATABASE_URL or settings.DATABASE_URL or ""
if raw_sync.startswith("postgresql+asyncpg://"):
    raw_sync = raw_sync.replace("postgresql+asyncpg://", "postgresql://", 1)
elif raw_sync.startswith("postgres://"):
    raw_sync = raw_sync.replace("postgres://", "postgresql://", 1)

if "?ssl=require" in raw_sync:
    raw_sync = raw_sync.replace("?ssl=require", "?sslmode=require")
elif "&ssl=require" in raw_sync:
    raw_sync = raw_sync.replace("&ssl=require", "&sslmode=require")

sync_engine = create_engine(
    raw_sync,
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
