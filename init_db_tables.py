import asyncio
from app.core.database import engine, init_db
from app.models.domain import Company, Document, FinancialMetric, DocumentChunk
from sqlalchemy import text

async def main():
    print("==================================================")
    print("🏗️ Initializing Financial RAG Relational & Vector Tables")
    print("==================================================")
    
    try:
        await init_db()
        print("✅ Database tables created successfully:")
        
        async with engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';
            """))
            tables = [row[0] for row in result.fetchall()]
            for table in sorted(tables):
                print(f"   • Table: {table}")
                
        print("\n🚀 Database schema is ready for document ingestion!")
    except Exception as e:
        print(f"❌ Schema creation failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())