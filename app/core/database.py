import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool
from app.core.config import get_settings

settings = get_settings()

# 1. Async Database Engine (asyncpg requires '?ssl=require')
raw_async_url = settings.NEON_DB_ASYNC_URL or settings.DATABASE_URL or ""
if raw_async_url.startswith("postgres://"):
    raw_async_url = raw_async_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif raw_async_url.startswith("postgresql://") and not raw_async_url.startswith("postgresql+asyncpg://"):
    raw_async_url = raw_async_url.replace("postgresql://", "postgresql+asyncpg://", 1)

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

def auto_migrate_schema():
    """Ensures newly added columns exist in Neon PostgreSQL without dropping data."""
    columns_to_verify = [
        ("documents", "file_hash", "VARCHAR(64)"),
        ("documents", "form_type", "VARCHAR(20) DEFAULT '10-K'"),
        ("documents", "fiscal_period", "VARCHAR(20)"),
        ("documents", "executive_summary", "TEXT"),
        ("documents", "key_risks", "TEXT"),
        ("documents", "growth_catalysts", "TEXT"),
        ("financial_metrics", "free_cash_flow", "FLOAT"),
        ("financial_metrics", "gross_margin", "FLOAT"),
        ("financial_metrics", "operating_margin", "FLOAT"),
        ("financial_metrics", "net_profit_margin", "FLOAT"),
        ("financial_metrics", "debt_to_equity", "FLOAT"),
        ("financial_metrics", "capital_expenditures", "FLOAT"),
        ("financial_metrics", "total_cash_and_equivalents", "FLOAT"),
        ("financial_metrics", "total_debt", "FLOAT"),
        ("financial_metrics", "shareholders_equity", "FLOAT")
    ]
    try:
        with sync_engine.connect() as conn:
            for table, col, col_type in columns_to_verify:
                try:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {col_type};"))
                    conn.commit()
                except Exception:
                    pass
    except Exception:
        pass

# Run automatic non-destructive column verification on startup
auto_migrate_schema()

async def init_db():
    async with async_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
