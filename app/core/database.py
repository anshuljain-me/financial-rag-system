from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool
from app.core.config import get_settings

settings = get_settings()

class Base(DeclarativeBase):
    pass


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'

# 1. Async Database Engine (asyncpg requires '?ssl=require' NOT '?sslmode=require')
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

async def init_db():
    """Initializes pgvector and keeps the deployed schema compatible with models."""
    import app.models.domain  # Registers all ORM models on Base.metadata
    async with async_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_repair_existing_schema)


def ensure_database_ready():
    """Synchronous startup hook for Streamlit before any query or ingestion runs."""
    import app.models.domain  # Registers all ORM models on Base.metadata
    with sync_engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        Base.metadata.create_all(bind=conn)
        _repair_existing_schema(conn)


def _repair_existing_schema(conn):
    """
    create_all() only creates missing tables; it does not migrate existing Neon
    tables. Keep older deployed tables insert-compatible with the current ORM.
    """
    metadata_tables = Base.metadata.tables

    for table in metadata_tables.values():
        table_name = table.name
        existing_columns = conn.execute(
            text("""
                SELECT column_name, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = :table_name
            """),
            {"table_name": table_name},
        ).mappings().all()

        if not existing_columns:
            continue

        existing_by_name = {row["column_name"]: row for row in existing_columns}
        model_column_names = {column.name for column in table.columns}

        for column in table.columns:
            if column.name in existing_by_name:
                continue

            column_type = column.type.compile(dialect=conn.dialect)
            conn.execute(
                text(
                    f"ALTER TABLE {_quote_ident(table_name)} "
                    f"ADD COLUMN IF NOT EXISTS {_quote_ident(column.name)} {column_type}"
                )
            )

        for column_name, info in existing_by_name.items():
            if column_name in model_column_names:
                continue
            if info["is_nullable"] == "NO" and info["column_default"] is None:
                conn.execute(
                    text(
                        f"ALTER TABLE {_quote_ident(table_name)} "
                        f"ALTER COLUMN {_quote_ident(column_name)} DROP NOT NULL"
                    )
                )
