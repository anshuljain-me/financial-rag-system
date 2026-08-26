import asyncio
from sqlalchemy import text
from app.core.database import engine, init_db
from app.core.config import get_settings
from google import genai

async def main():
    print("==================================================")
    print("🔍 Testing Financial RAG Environment & Connectivity")
    print("==================================================")
    
    # 1. Check Configuration Settings
    print("\n1. Verifying Settings from .env...")
    settings = get_settings()
    print(f"   • Project: {settings.PROJECT_NAME}")
    print(f"   • Database Host: {settings.DATABASE_URL.split('@')[-1].split('/')[0]}")
    print(f"   • Target LLM: {settings.GEMINI_FAST_MODEL}")
    
    # 2. Test Neon PostgreSQL & pgvector
    print("\n2. Connecting to Neon PostgreSQL & Checking pgvector...")
    try:
        await init_db()
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector';"))
            ext = result.scalar()
            if ext == "vector":
                print("   ✅ PostgreSQL connected successfully.")
                print("   ✅ pgvector extension is ACTIVE and ready for vector embeddings.")
            else:
                print("   ⚠️ Connected to PostgreSQL, but pgvector extension was not detected.")
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        return

    # 3. Test Google Gemini AI Generation
    print("\n3. Testing Google Gemini Generation...")
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY.strip().strip("'").strip('"'))
        response = client.models.generate_content(
            model=settings.GEMINI_FAST_MODEL,
            contents="Ping. Respond with 'Financial RAG Engine Ready' only."
        )
        print(f"   ✅ Gemini API connected successfully ({settings.GEMINI_FAST_MODEL}).")
        print(f"   • LLM Response: {response.text.strip()}")
    except Exception as e:
        print(f"   ❌ Gemini API test failed: {e}")
        return

    print("\n==================================================")
    print("🚀 All Core Services Verified Successfully!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())