import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy import select, and_, delete

from app.core.database import AsyncSessionLocal
from app.models.domain import Company, Document, FinancialMetric, DocumentChunk
from app.ingestion.hasher import compute_sha256
from app.ingestion.parser import parse_pdf_structure
from app.extraction.kpi_extractor import FinancialExtractor
from app.rag.embedder import FinancialEmbedder

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Robust End-to-End SEC 10-K Ingestion Pipeline:
    1. Cryptographic SHA-256 deduplication
    2. Structure-aware multi-column PDF table parsing
    3. Single-pass KPI & strategic summary extraction
    4. Safe type serialization (JSON strings for Text columns)
    5. Clean relational and 768-D vector chunk indexing
    """

    def __init__(self):
        self.extractor = FinancialExtractor()
        self.embedder = FinancialEmbedder()

    async def process_file(self, pdf_path: Path, ticker_override: Optional[str] = None) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        file_hash = compute_sha256(pdf_path)

        async with AsyncSessionLocal() as session:
            # 1. Check for duplicate by file hash
            stmt = select(Document).where(Document.file_hash == file_hash)
            res = await session.execute(stmt)
            existing_doc = res.scalar_one_or_none()

            if existing_doc:
                return {
                    "status": "cached",
                    "document_id": str(existing_doc.id),
                    "ticker": existing_doc.ticker,
                    "fiscal_year": existing_doc.fiscal_year,
                    "message": f"Document {pdf_path.name} already indexed."
                }

            # 2. Parse PDF Structure & Tables
            parsed_data = parse_pdf_structure(pdf_path)
            full_text_sample = "\n\n".join([page.get("text", "") for page in parsed_data.get("pages", [])])

            # 3. Extract Structured Financial Line Items & Executive Summary
            extracted = self.extractor.extract_kpis_and_summary(full_text_sample)
            kpis = extracted.get("kpis", {})
            ratios = extracted.get("calculated_ratios", {})
            summary = extracted.get("summary", {})

            ticker = (ticker_override or kpis.get("ticker") or "UNKNOWN").upper()
            company_name = kpis.get("company_name") or f"{ticker} Inc."
            fiscal_year = int(kpis.get("fiscal_year") or 2025)
            fiscal_period = f"FY{fiscal_year}"

            # 4. Resolve Company in Database
            stmt_comp = select(Company).where(Company.ticker == ticker)
            res_comp = await session.execute(stmt_comp)
            company = res_comp.scalar_one_or_none()

            if not company:
                company = Company(
                    ticker=ticker,
                    company_name=company_name,
                    sector=kpis.get("sector", "General"),
                    industry=kpis.get("industry", "Diversified")
                )
                session.add(company)
                await session.flush()

            # 5. Clean up any existing duplicate document for this specific year
            stmt_old_doc = select(Document).where(
                and_(
                    Document.company_id == company.id,
                    Document.fiscal_year == fiscal_year,
                    Document.form_type == "10-K"
                )
            )
            res_old = await session.execute(stmt_old_doc)
            old_doc = res_old.scalar_one_or_none()
            if old_doc:
                await session.delete(old_doc)
                await session.flush()

            # 6. Serialize Lists to Safe Strings for Text Columns (Prevents DBAPIError)
            key_risks_val = summary.get("key_risks", [])
            key_risks_str = json.dumps(key_risks_val) if isinstance(key_risks_val, (list, dict)) else str(key_risks_val or "")

            growth_cat_val = summary.get("growth_catalysts", [])
            growth_cat_str = json.dumps(growth_cat_val) if isinstance(growth_cat_val, (list, dict)) else str(growth_cat_val or "")

            doc = Document(
                company_id=company.id,
                ticker=ticker,
                form_type="10-K",
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                file_path=str(pdf_path),
                file_hash=file_hash,
                executive_summary=summary.get("executive_summary", ""),
                key_risks=key_risks_str,
                growth_catalysts=growth_cat_str
            )
            session.add(doc)
            await session.flush()

            # 7. Create Financial Metric with Safe Float Conversion
            def safe_float(v):
                try:
                    return float(v) if v is not None and str(v).strip() != "" else None
                except Exception:
                    return None

            metric = FinancialMetric(
                document_id=doc.id,
                revenue=safe_float(kpis.get("revenue")),
                gross_profit=safe_float(kpis.get("gross_profit")),
                operating_income=safe_float(kpis.get("operating_income")),
                net_income=safe_float(kpis.get("net_income")),
                diluted_eps=safe_float(kpis.get("diluted_eps")),
                gross_margin=safe_float(ratios.get("gross_margin")),
                operating_margin=safe_float(ratios.get("operating_margin")),
                net_profit_margin=safe_float(ratios.get("net_profit_margin")),
                free_cash_flow=safe_float(ratios.get("free_cash_flow")),
                total_debt=safe_float(kpis.get("total_debt")),
                total_cash_and_equivalents=safe_float(kpis.get("total_cash_and_equivalents")),
                debt_to_equity=safe_float(ratios.get("debt_to_equity"))
            )
            session.add(metric)

            # 8. Create Structure-Preserved Document Chunks with 768-D Embeddings
            chunks_to_add = []
            chunk_idx = 0

            for page in parsed_data.get("pages", []):
                p_num = page.get("page_number", 1)
                p_text = page.get("text", "").strip()
                p_section = page.get("detected_section") or "Item 8 Consolidated Financial Statements"

                if not p_text:
                    continue

                emb = self.embedder.embed_text(p_text)

                chunk = DocumentChunk(
                    document_id=doc.id,
                    ticker=ticker,
                    section=p_section,
                    chunk_type="table" if "---" in p_text or "|" in p_text else "text",
                    content=p_text,
                    page_number=p_num,
                    chunk_index=chunk_idx,
                    embedding=emb
                )
                chunks_to_add.append(chunk)
                chunk_idx += 1

            if chunks_to_add:
                session.add_all(chunks_to_add)

            # Commit the entire transaction
            await session.commit()

            return {
                "status": "success",
                "document_id": str(doc.id),
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "chunks_indexed": len(chunks_to_add),
                "company_name": company_name
            }
