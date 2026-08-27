import os
import json
import logging
import asyncio
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy import select, and_

from app.core.database import AsyncSessionLocal
from app.models.domain import Company, Document, FinancialMetric, DocumentChunk
from app.ingestion.hasher import calculate_file_hash
from app.ingestion.parser import StructureAwarePDFParser
from app.extraction.kpi_extractor import FinancialExtractor
from app.rag.embedder import FinancialEmbedder

logger = logging.getLogger(__name__)


def _clean_text(value: Any, default: str = "", max_length: Optional[int] = None) -> str:
    if value is None:
        value = default
    if isinstance(value, (list, dict)):
        value = json.dumps(value)
    cleaned = str(value).strip()
    if not cleaned:
        cleaned = default
    if max_length:
        cleaned = cleaned[:max_length]
    return cleaned


def _clean_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        cleaned = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(cleaned) or math.isinf(cleaned):
        return None
    return cleaned

class IngestionPipeline:
    """
    Robust Production SEC Filing Ingestion & Vector Indexing Pipeline.
    Handles duplicate replacement, atomic transactions, vector embedding, and ratio computation.
    """

    def __init__(self):
        self.parser = StructureAwarePDFParser()
        self.extractor = FinancialExtractor()
        self.embedder = FinancialEmbedder()

    def process_file_sync(self, pdf_path: Path, ticker_override: Optional[str] = None) -> Dict[str, Any]:
        return asyncio.run(self.process_file(pdf_path, ticker_override=ticker_override))

    async def process_file(self, pdf_path: Path, ticker_override: Optional[str] = None) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found at: {pdf_path}")

        file_hash = calculate_file_hash(pdf_path)

        # 1. Parse PDF with structure awareness
        parsed_doc = self.parser.parse_pdf(pdf_path)
        combined_sample_text = "\n\n".join([c["text"] for c in parsed_doc["chunks"][:8]])

        # 2. Extract structured financial data
        extracted = self.extractor.extract_kpis_and_summary(combined_sample_text)
        kpis = extracted["kpis"]
        ratios = extracted["calculated_ratios"]
        summary = extracted["summary"]

        ticker = _clean_text(ticker_override or kpis.get("ticker"), "UNKNOWN", 10).upper()
        company_name = _clean_text(kpis.get("company_name"), f"{ticker} Inc.", 255)
        fiscal_year = int(kpis.get("fiscal_year") or 2025)
        fiscal_period = _clean_text(kpis.get("fiscal_period"), f"FY{fiscal_year}", 20)
        form_type = "10-K" if "10-K" in pdf_path.name or "FY" in fiscal_period else "10-Q"

        async with AsyncSessionLocal() as session:
            # 3. Get or Create Company (prevent unique violation)
            stmt = select(Company).where(Company.ticker == ticker)
            res = await session.execute(stmt)
            company = res.scalars().first()

            if not company:
                company = Company(
                    ticker=ticker,
                    company_name=company_name
                )
                session.add(company)
                await session.flush()

            # 4. Clean up any existing document for this exact (ticker, fiscal_year, form_type)
            del_stmt = select(Document).where(
                and_(
                    Document.ticker == ticker,
                    Document.fiscal_year == fiscal_year,
                    Document.form_type == form_type
                )
            )
            existing_docs_res = await session.execute(del_stmt)
            existing_docs = existing_docs_res.scalars().all()
            for old_doc in existing_docs:
                await session.delete(old_doc)
            await session.flush()

            # 5. Create new Document
            doc = Document(
                company_id=company.id,
                ticker=ticker,
                form_type=form_type,
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                file_hash=file_hash,
                executive_summary=_clean_text(summary.get("executive_summary")),
                key_risks=_clean_text(summary.get("key_risks"), "[]"),
                growth_catalysts=_clean_text(summary.get("growth_catalysts"), "[]")
            )
            session.add(doc)
            await session.flush()

            # 6. Create FinancialMetric Record
            metric = FinancialMetric(
                document_id=doc.id,
                revenue=_clean_float(kpis.get("revenue")),
                gross_profit=_clean_float(kpis.get("gross_profit")),
                operating_income=_clean_float(kpis.get("operating_income")),
                net_income=_clean_float(kpis.get("net_income")),
                diluted_eps=_clean_float(kpis.get("diluted_eps")),
                operating_cash_flow=_clean_float(kpis.get("operating_cash_flow")),
                capital_expenditures=_clean_float(kpis.get("capital_expenditures")),
                free_cash_flow=_clean_float(ratios.get("free_cash_flow")),
                total_cash_and_equivalents=_clean_float(kpis.get("total_cash_and_equivalents")),
                total_debt=_clean_float(kpis.get("total_debt")),
                shareholders_equity=_clean_float(kpis.get("shareholders_equity")),
                gross_margin=_clean_float(ratios.get("gross_margin")),
                operating_margin=_clean_float(ratios.get("operating_margin")),
                net_profit_margin=_clean_float(ratios.get("net_profit_margin")),
                debt_to_equity=_clean_float(ratios.get("debt_to_equity"))
            )
            session.add(metric)

            # 7. Create DocumentChunk Records with Vector Embeddings
            for c in parsed_doc["chunks"]:
                raw_content = c.get("text", "").strip()
                if not raw_content:
                    continue

                emb = self.embedder.embed_text(raw_content)

                chunk = DocumentChunk(
                    document_id=doc.id,
                    ticker=ticker,
                    section=_clean_text(c.get("section"), "SEC Disclosures", 50),
                    chunk_type=_clean_text(c.get("type"), "text", 20),
                    content=raw_content,
                    page_number=c.get("page", 1),
                    chunk_index=c.get("chunk_index", 0),
                    embedding=emb
                )
                session.add(chunk)

            await session.commit()

        return {
            "status": "success",
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "chunks_count": len(parsed_doc["chunks"]),
            "revenue": kpis.get("revenue"),
            "gross_margin": ratios.get("gross_margin")
        }
