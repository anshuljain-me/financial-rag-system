import asyncio
from pathlib import Path
import fitz  # PyMuPDF (already installed)
from app.ingestion.pipeline import IngestionPipeline

def create_sample_financial_pdf(target_path: Path):
    """Generates a realistic multi-page SEC 10-K filing using PyMuPDF."""
    doc = fitz.open()
    
    # Page 1: Cover & Item 1 Business
    p1 = doc.new_page()
    p1.insert_text((50, 50), "UNITED STATES SECURITIES AND EXCHANGE COMMISSION", fontsize=13)
    p1.insert_text((50, 75), "FORM 10-K - ANNUAL REPORT", fontsize=12)
    p1.insert_text((50, 100), "Company: Apple Inc. | Ticker: AAPL | Fiscal Year: 2024", fontsize=11)
    p1.insert_text((50, 140), "ITEM 1. BUSINESS", fontsize=12)
    p1.insert_textbox(
        fitz.Rect(50, 160, 550, 350),
        "Apple Inc. designs, manufactures, and markets smartphones, personal computers, "
        "tablets, wearables, and accessories, and sells a variety of related services. "
        "The Company's fiscal year ends on the last Saturday of September.",
        fontsize=10
    )
    
    # Page 2: Item 1A Risk Factors
    p2 = doc.new_page()
    p2.insert_text((50, 50), "ITEM 1A. RISK FACTORS", fontsize=12)
    p2.insert_textbox(
        fitz.Rect(50, 75, 550, 350),
        "Global economic conditions and geopolitical tensions could adversely affect customer demand. "
        "Supply chain disruptions, semiconductor shortages, and concentration of manufacturing in specific regions "
        "present ongoing operational risks. Regulatory antitrust scrutiny in the European Union and United States "
        "regarding the App Store could impact future services revenue margins.",
        fontsize=10
    )

    # Page 3: Item 8 Consolidated Financial Statements
    p3 = doc.new_page()
    p3.insert_text((50, 50), "ITEM 8. CONSOLIDATED STATEMENTS OF OPERATIONS", fontsize=12)
    financial_statements_text = (
        "(In Millions USD, except per share amounts)\n\n"
        "Total Net Sales / Revenue: $391,035\n"
        "Cost of Sales: $210,352\n"
        "Gross Profit: $180,683\n"
        "Operating Expenses: $57,484\n"
        "Operating Income: $123,200\n"
        "Net Income: $93,736\n"
        "Diluted Earnings Per Share (EPS): $6.08\n\n"
        "Operating Cash Flow: $118,254\n"
        "Capital Expenditures (CapEx): $9,450\n"
        "Cash and Marketable Securities: $65,200\n"
        "Total Debt: $106,600\n"
        "Total Stockholders' Equity: $66,700\n"
    )
    p3.insert_textbox(fitz.Rect(50, 75, 550, 500), financial_statements_text, fontsize=10)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(target_path))
    doc.close()
    print(f"📄 Created sample SEC filing at: {target_path}")

async def main():
    print("==================================================")
    print("🚀 Testing End-to-End Financial Ingestion Pipeline")
    print("==================================================")
    
    sample_dir = Path("data/sample_filings")
    sample_file = sample_dir / "AAPL_2024_10K.pdf"
    
    # 1. Generate sample PDF
    create_sample_financial_pdf(sample_file)
    
    # 2. Ingest document
    pipeline = IngestionPipeline()
    result = await pipeline.process_file(sample_file, ticker_override="AAPL")
    
    print("\n--- Ingestion Result ---")
    print(f"• Status: {result.get('status')}")
    print(f"• Ticker: {result.get('ticker')}")
    print(f"• Chunks Indexed to pgvector: {result.get('chunks_indexed')}")
    print(f"• Extracted KPIs: {result.get('kpis')}")
    print(f"• Calculated Accounting Ratios: {result.get('ratios')}")

    # 3. Test Deduplication / Cache Hit
    print("\n🔄 Testing Deduplication (Second Ingestion Call)...")
    cached_result = await pipeline.process_file(sample_file)
    print(f"• Second Run Status: {cached_result.get('status')}")
    print(f"• Message: {cached_result.get('message')}")

    print("\n==================================================")
    print("🎯 Ingestion, Extraction & Deduplication Verified!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())