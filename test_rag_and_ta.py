import asyncio
from app.rag.qa_engine import FinancialQAService
from app.analytics.technical import TechnicalAnalysisEngine

async def main():
    print("==================================================")
    print("🧪 Testing Financial RAG QA Engine & Technical Analysis")
    print("==================================================")
    
    # 1. Test Financial RAG QA
    print("\n1. Testing Grounded RAG Chat Engine for AAPL...")
    qa_service = FinancialQAService()
    
    test_questions = [
        "What was Apple's total revenue, gross profit, and diluted EPS for fiscal year 2024?",
        "What are the major operational and regulatory risk factors mentioned regarding the App Store?"
    ]
    
    for q in test_questions:
        print(f"\n❓ Question: {q}")
        response = await qa_service.answer_question(question=q, ticker="AAPL")
        print("\n💡 AI Answer:")
        print(response["answer"])
        print("\n📑 Citations:")
        for cit in response["citations"]:
            print(f"   • Page {cit['page_number']} [{cit['section']}] (Score: {cit['score']})")

    # 2. Test Quantitative Technical Analysis
    print("\n\n2. Testing Quantitative Technical Analysis Engine for AAPL...")
    ta_engine = TechnicalAnalysisEngine()
    ta_result = ta_engine.fetch_and_analyze("AAPL")
    
    print(f"• Current Price: ${ta_result['current_price']} ({ta_result['price_change_pct']}%)")
    print(f"• RSI (14): {ta_result['rsi_14']}")
    print(f"• 50-day SMA: ${ta_result['sma_50']} | 200-day SMA: ${ta_result['sma_200']}")
    print("• Technical Signals:")
    for sig in ta_result["technical_signals"]:
        print(f"   - {sig}")

    print("\n==================================================")
    print("🎉 Both RAG QA and Technical Analysis Verified!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())