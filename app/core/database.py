import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool
from app.core.config import get_settings

settings = get_settings()

# 1. Async Database Engine
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

# 2. Synchronous Database Engine
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

SCHEMA_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS companies (
    id UUID PRIMARY KEY,
    ticker VARCHAR(10) UNIQUE NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(100),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    company_id UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    form_type VARCHAR(20) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_period VARCHAR(20) NOT NULL,
    filing_date VARCHAR(50),
    file_path VARCHAR(500),
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    executive_summary TEXT,
    key_risks TEXT,
    growth_catalysts TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS financial_metrics (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,
    revenue DOUBLE PRECISION,
    gross_profit DOUBLE PRECISION,
    operating_income DOUBLE PRECISION,
    net_income DOUBLE PRECISION,
    diluted_eps DOUBLE PRECISION,
    gross_margin DOUBLE PRECISION,
    operating_margin DOUBLE PRECISION,
    net_profit_margin DOUBLE PRECISION,
    free_cash_flow DOUBLE PRECISION,
    total_debt DOUBLE PRECISION,
    total_cash_and_equivalents DOUBLE PRECISION,
    debt_to_equity DOUBLE PRECISION,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    section VARCHAR(50),
    chunk_type VARCHAR(20) DEFAULT 'text',
    content TEXT NOT NULL,
    page_number INTEGER,
    chunk_index INTEGER,
    embedding vector(768),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_ticker_section ON document_chunks (ticker, section);
CREATE INDEX IF NOT EXISTS idx_documents_ticker ON documents (ticker);
CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies (ticker);
"""

async def init_db():
    """Explicitly initializes database extension, tables, and indexes in PostgreSQL."""
    async with async_engine.begin() as conn:
        for statement in SCHEMA_DDL.strip().split(";"):
            stmt_clean = statement.strip()
            if stmt_clean:
                try:
                    await conn.execute(text(stmt_clean))
                except Exception:
                    pass
