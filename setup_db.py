import asyncio
from app.core.database import init_db, sync_engine, text

async def main():
    print("🚀 Initializing PostgreSQL schema (tables, vector extension, and indexes)...")
    await init_db()
    with sync_engine.connect() as conn:
        res = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"))
        tables = [row[0] for row in res.fetchall()]
        print(f"✅ Verified Tables in PostgreSQL: {tables}")

if __name__ == "__main__":
    asyncio.run(main())
