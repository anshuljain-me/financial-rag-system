from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.core.config import get_settings

settings = get_settings()

# Normalize database URL for asyncpg
db_url = settings.DATABASE_URL
if "sslmode=require" in db_url:
    db_url = db_url.replace("sslmode=require", "ssl=require")
elif "?" not in db_url and "neon.tech" in db_url:
    db_url = f"{db_url}?ssl=require"

# NullPool ensures asyncpg connections are never leaked across different asyncio event loops
engine = create_async_engine(
    db_url,
    echo=False,
    future=True,
    poolclass=NullPool
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# Synchronous Engine & Session (Thread-safe for Streamlit UI)
sync_db_url = settings.SYNC_DATABASE_URL
if "postgresql+asyncpg://" in sync_db_url:
    sync_db_url = sync_db_url.replace("postgresql+asyncpg://", "postgresql://")

sync_engine = create_engine(
    sync_db_url,
    echo=False,
    pool_pre_ping=True
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False
)

class Base(DeclarativeBase):
    pass

async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
