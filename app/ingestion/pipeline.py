import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid
from sqlalchemy import select, and_

from app.core.database import AsyncSessionLocal
from app.models.domain import Company, Document, FinancialMetric, DocumentChunk
from app.ingestion.parser import StructureAwarePDFParser
from app.extraction.kpi_extractor import FinancialExtractor
from app.rag.embedder import FinancialEmbedder
from app.ingestion.hasher import compute_file_hash

class IngestionPipeline:
    """
    Unified Ingestion & Indexing Pipeline:
    1. Cryptographic SHA-256 Deduplication Hashing
    2. Structure-Aware PDF & Table Parsing (PyMuPDF)
    3. Single-Pass KPI & Executive Summary Extraction (Gemini)
    4. 768-D Vector Embeddings & pgvector Chunk Indexing
    """

    def __init__(self):
        self.parser = StructureAwarePDFParser()
        self.extractor = FinancialExtractor()
        self.embedder = FinancialEmbedder()

    async def process_file(self, pdf_path: Path, ticker_override: Optional[str] = None) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return {"error": f"File not found: {pdf_path}"}

        file_hash = compute_file_hash(pdf_path)

        async with AsyncSessionLocal() as session:
            # 1. SHA-256 Deduplication Check
            stmt = select(Document).where(Document.file_hash == file_hash)
            res = await session.execute(stmt)
            existing_doc = res.scalar_one_or_none()

            if existing_doc:
                return {
                    "status": "cached",
                    "document_id": str(existing_doc.id),
                    "ticker": existing_doc.ticker,
                    "fiscal_year": existing_doc.fiscal_year,
                    "message": "Filing already processed and indexed."
                }

            # 2. Parse PDF Structure
            pages_data = self.parser.extract_text_and_tables(pdf_path)
            full_text = "\n\n".join([p["text"] for p in pages_data])

            # 3. Extract KPIs and Summary
            extracted_bundle = self.extractor.extract_kpis_and_summary(full_text[:40000])
            kpis = extracted_bundle.get("kpis", {})
            ratios = extracted_bundle.get("calculated_ratios", {})
            summary = extracted_bundle.get("summary", {})

            ticker = ticker_override.upper() if ticker_override else kpis.get("ticker", "TICKER").upper()
            fiscal_year = kpis.get("fiscal_year", 2025)
            fiscal_period = kpis.get("fiscal_period", f"FY{fiscal_year}")

            # 4. Get or Create Company
            comp_stmt = select(Company).where(Company.ticker == ticker)
            comp_res = await session.execute(comp_stmt)
            company = comp_res.scalar_one_or_none()

            if not company:
                company = Company(
                    ticker=ticker,
                    company_name=kpis.get("company_name", f"{ticker} Inc.")
                )
                session.add(company)
                await session.flush()

            # 5. Create Document Record
            document = Document(
                company_id=company.id,
                ticker=ticker,
                form_type="10-K",
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                file_hash=file_hash,
                file_path=str(pdf_path),
                executive_summary=summary.get("executive_summary", ""),
                key_risks=summary.get("key_risks", []),
                growth_catalysts=summary.get("growth_catalysts", [])
            )
            session.add(document)
            await session.flush()

            # 6. Create Financial Metrics Record
            metric = FinancialMetric(
                document_id=document.id,
                revenue=kpis.get("revenue"),
                gross_profit=kpis.get("gross_profit"),
                operating_income=kpis.get("operating_income"),
                net_income=kpis.get("net_income"),
                diluted_eps=kpis.get("diluted_eps"),
                gross_margin=ratios.get("gross_margin"),
                operating_margin=ratios.get("operating_margin"),
                net_profit_margin=ratios.get("net_profit_margin"),
                operating_cash_flow=kpis.get("operating_cash_flow"),
                capital_expenditures=kpis.get("capital_expenditures"),
                free_cash_flow=ratios.get("free_cash_flow"),
                total_cash_and_equivalents=kpis.get("total_cash_and_equivalents"),
                total_debt=kpis.get("total_debt"),
                shareholders_equity=kpis.get("shareholders_equity"),
                debt_to_equity=ratios.get("debt_to_equity")
            )
            session.add(metric)

            # 7. Generate Chunks and Vector Embeddings
            for p_idx, page in enumerate(pages_data):
                p_text = page.get("text", "").strip()
                if not p_text:
                    continue

                emb = self.embedder.embed_text(p_text)
                chunk = DocumentChunk(
                    document_id=document.id,
                    ticker=ticker,
                    section=page.get("section", "SEC Disclosures"),
                    chunk_type="text",
                    content=p_text,
                    page_number=page.get("page_number", p_idx + 1),
                    chunk_index=p_idx,
                    embedding=emb
                )
                session.add(chunk)

            await session.commit()

            return {
                "status": "success",
                "document_id": str(document.id),
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "message": f"Successfully ingested and indexed {ticker} ({fiscal_period})!"
            }
