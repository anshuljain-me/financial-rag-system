import asyncio
from app.core.database import SyncSessionLocal
from app.models.domain import Company, Document, DocumentChunk
from app.rag.qa_engine import FinancialQAService
from sqlalchemy import select, func

async def main():
    print("==================================================")
    print("🔍 Diagnosing Deep-Dive RAG Chat Engine")
    print("==================================================")

    # 1. Check Chunks in PostgreSQL
    print("\n1. Checking Indexed Chunks in Database...")
    with SyncSessionLocal() as session:
        stmt = select(DocumentChunk.ticker, func.count(DocumentChunk.id)).group_by(DocumentChunk.ticker)
        res = session.execute(stmt).all()
        if not res:
            print("   ⚠️ No chunks found in the database. Please ingest a company first.")
            return
        for t, count in res:
            print(f"   • Ticker: {t} -> {count} vector chunks indexed.")

    # 2. Test RAG Retrieval & Gemini Generation
    sample_ticker = res[0][0]
    print(f"\n2. Testing Live Chat Generation for '{sample_ticker}'...")
    qa = FinancialQAService()
    try:
        response = await qa.answer_question(
            question=f"What are the primary operational risks and financial metrics for {sample_ticker}?",
            ticker=sample_ticker
        )
        print(f"   ✅ SUCCESS! Response received ({len(response['answer'])} chars):")
        print(f"   • Answer Preview: {response['answer'][:180]}...")
        print(f"   • Citations Count: {len(response['citations'])}")
    except Exception as e:
        print(f"   ❌ QA Generation Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
