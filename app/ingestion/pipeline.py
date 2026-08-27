import asyncio
import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from sqlalchemy import select, and_
from app.core.database import AsyncSessionLocal, init_db
from app.models.domain import Company, Document, FinancialMetric, DocumentChunk
from app.ingestion.hasher import Hasher
from app.ingestion.parser import SECParser
from app.extraction.kpi_extractor import FinancialExtractor
from app.rag.embedder import FinancialEmbedder

class IngestionPipeline:
    """
    Production Ingestion Pipeline:
    1. Cryptographic SHA-256 Deduplication.
    2. Structure-Aware PyMuPDF Table Parsing.
    3. Single-Pass KPI & Executive Summary Extraction.
    4. Auto-Table Initialization & Self-Healing Database Upserts.
    5. 768-D Vector Embeddings & pgvector chunk indexing.
    """

    def __init__(self):
        self.parser = SECParser()
        self.extractor = FinancialExtractor()
        self.embedder = FinancialEmbedder()

    async def process_file(self, pdf_path: Path, ticker_override: Optional[str] = None) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Filing not found at: {pdf_path}")

        # Auto-create tables and vector extension if not present
        try:
            await init_db()
        except Exception:
            pass

        file_hash = Hasher.compute_file_hash(pdf_path)

        async with AsyncSessionLocal() as session:
            # 1. Check if exact hash is already indexed
            stmt = select(Document).where(Document.file_hash == file_hash)
            existing_doc = (await session.execute(stmt)).scalars().first()
            if existing_doc:
                return {
                    "status": "cached",
                    "document_id": str(existing_doc.id),
                    "ticker": existing_doc.ticker,
                    "period": existing_doc.fiscal_period
                }

            # 2. Parse PDF structure and tables
            parsed_data = self.parser.parse_pdf(pdf_path)
            chunks = parsed_data.get("chunks", [])
            raw_text = parsed_data.get("full_text", "")
            meta = parsed_data.get("metadata", {})

            ticker = (ticker_override or meta.get("ticker", "TICKER")).upper()
            form_type = meta.get("form_type", "10-K")
            fiscal_year = meta.get("fiscal_year", 2025)
            fiscal_period = meta.get("fiscal_period", f"FY{fiscal_year}")

            # 3. Single-Pass KPI Extraction
            extraction_res = self.extractor.extract_kpis_and_summary(raw_text[:40000])
            kpis = extraction_res.get("kpis", {})
            ratios = extraction_res.get("calculated_ratios", {})
            summary_dict = extraction_res.get("summary", {})

            company_name = kpis.get("company_name") or meta.get("company_name") or f"{ticker} Inc."

            # 4. Get or Create Company
            comp_stmt = select(Company).where(Company.ticker == ticker)
            company = (await session.execute(comp_stmt)).scalars().first()
            if not company:
                company = Company(
                    id=str(uuid.uuid4()),
                    ticker=ticker,
                    company_name=company_name
                )
                session.add(company)
                await session.flush()

            # 5. Clean up any existing document for the exact same ticker & period to prevent collision
            period_stmt = select(Document).where(
                and_(
                    Document.ticker == ticker,
                    Document.form_type == form_type,
                    Document.fiscal_year == fiscal_year,
                    Document.fiscal_period == fiscal_period
                )
            )
            old_doc = (await session.execute(period_stmt)).scalars().first()
            if old_doc:
                await session.delete(old_doc)
                await session.flush()

            # 6. Create Document Record
            doc_id = str(uuid.uuid4())
            doc = Document(
                id=doc_id,
                company_id=company.id,
                ticker=ticker,
                form_type=form_type,
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                file_path=str(pdf_path),
                file_hash=file_hash,
                executive_summary=summary_dict.get("executive_summary", ""),
                key_risks=json.dumps(summary_dict.get("key_risks", [])),
                growth_catalysts=json.dumps(summary_dict.get("growth_catalysts", []))
            )
            session.add(doc)
            await session.flush()

            # 7. Create Financial Metric Record
            metric = FinancialMetric(
                id=str(uuid.uuid4()),
                document_id=doc.id,
                revenue=kpis.get("revenue"),
                gross_profit=kpis.get("gross_profit"),
                operating_income=kpis.get("operating_income"),
                net_income=kpis.get("net_income"),
                diluted_eps=kpis.get("diluted_eps"),
                operating_cash_flow=kpis.get("operating_cash_flow"),
                capital_expenditures=kpis.get("capital_expenditures"),
                free_cash_flow=kpis.get("free_cash_flow") or ratios.get("free_cash_flow"),
                total_cash_and_equivalents=kpis.get("total_cash_and_equivalents"),
                total_debt=kpis.get("total_debt"),
                shareholders_equity=kpis.get("shareholders_equity"),
                gross_margin=ratios.get("gross_margin"),
                operating_margin=ratios.get("operating_margin"),
                net_profit_margin=ratios.get("net_profit_margin"),
                debt_to_equity=ratios.get("debt_to_equity")
            )
            session.add(metric)

            # 8. Index Chunks & Embeddings
            for c_dict in chunks:
                c_text = c_dict.get("content", "")
                if not c_text.strip():
                    continue
                emb = self.embedder.embed_text(c_text)
                chunk_record = DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=doc.id,
                    ticker=ticker,
                    section=c_dict.get("section", "SEC Disclosures"),
                    chunk_type=c_dict.get("chunk_type", "text"),
                    content=c_text,
                    page_number=c_dict.get("page_number", 1),
                    chunk_index=c_dict.get("chunk_index", 0),
                    embedding=emb
                )
                session.add(chunk_record)

            await session.commit()

            return {
                "status": "success",
                "document_id": str(doc.id),
                "ticker": ticker,
                "period": fiscal_period,
                "chunks_indexed": len(chunks)
            }
