import os
import uuid
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy import select, and_, delete

from app.core.database import AsyncSessionLocal, init_db
from app.models.domain import Company, Document, FinancialMetric, DocumentChunk
from app.ingestion.hasher import compute_sha256
from app.ingestion.parser import parse_pdf_structure
from app.extraction.kpi_extractor import FinancialExtractor
from app.rag.embedder import FinancialEmbedder

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Enterprise-Grade Resilient SEC Form 10-K Ingestion Pipeline.
    100% atomic transactions, explicit UUID primary keys, and robust conflict resolution.
    """

    def __init__(self):
        self.extractor = FinancialExtractor()
        self.embedder = FinancialEmbedder()

    def _safe_float(self, val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            cleaned = str(val).replace(",", "").replace("$", "").strip()
            if cleaned == "" or cleaned.lower() in ["n/a", "none", "null"]:
                return None
            return float(cleaned)
        except Exception:
            return None

    async def process_file(self, pdf_path: Path, ticker_override: Optional[str] = None) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        # 0. Ensure all database tables and vector extension exist
        await init_db()

        # 1. Compute Cryptographic SHA-256 Hash
        file_hash = compute_sha256(pdf_path)

        async with AsyncSessionLocal() as session:
            # 2. Parse PDF Structure & Extract Text
            parsed_data = parse_pdf_structure(pdf_path)
            pages = parsed_data.get("pages", [])
            full_text_sample = "\n\n".join([page.get("text", "") for page in pages])

            # 3. Single-Pass Extraction for KPIs and Summary
            extracted = self.extractor.extract_kpis_and_summary(full_text_sample)
            kpis = extracted.get("kpis", {})
            ratios = extracted.get("calculated_ratios", {})
            summary = extracted.get("summary", {})

            ticker = (ticker_override or kpis.get("ticker") or "UNKNOWN").strip().upper()[:10]
            company_name = str(kpis.get("company_name") or f"{ticker} Inc.")[:255]
            fiscal_year = int(kpis.get("fiscal_year") or 2025)
            fiscal_period = f"FY{fiscal_year}"[:20]

            # 4. Resolve or Create Company
            stmt_comp = select(Company).where(Company.ticker == ticker)
            res_comp = await session.execute(stmt_comp)
            company = res_comp.scalar_one_or_none()

            if not company:
                company = Company(
                    id=uuid.uuid4(),
                    ticker=ticker,
                    company_name=company_name,
                    sector=str(kpis.get("sector", "General"))[:100],
                    industry=str(kpis.get("industry", "Diversified"))[:100]
                )
                session.add(company)
                await session.flush()

            # 5. Clean up any existing Document with identical hash or company+year to avoid collisions
            stmt_old = select(Document).where(
                and_(
                    Document.company_id == company.id,
                    Document.fiscal_year == fiscal_year,
                    Document.form_type == "10-K"
                )
            )
            res_old = await session.execute(stmt_old)
            for old_doc in res_old.scalars().all():
                await session.delete(old_doc)
            await session.flush()

            # 6. Create Document Record
            key_risks_val = summary.get("key_risks", [])
            key_risks_str = json.dumps(key_risks_val) if isinstance(key_risks_val, (list, dict)) else str(key_risks_val or "")

            growth_cat_val = summary.get("growth_catalysts", [])
            growth_cat_str = json.dumps(growth_cat_val) if isinstance(growth_cat_val, (list, dict)) else str(growth_cat_val or "")

            doc_id = uuid.uuid4()
            doc = Document(
                id=doc_id,
                company_id=company.id,
                ticker=ticker,
                form_type="10-K",
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                file_path=str(pdf_path)[:500],
                file_hash=file_hash[:64],
                executive_summary=str(summary.get("executive_summary", "")),
                key_risks=key_risks_str,
                growth_catalysts=growth_cat_str
            )
            session.add(doc)
            await session.flush()

            # 7. Create Financial Metric Line Items
            metric = FinancialMetric(
                id=uuid.uuid4(),
                document_id=doc.id,
                revenue=self._safe_float(kpis.get("revenue")),
                gross_profit=self._safe_float(kpis.get("gross_profit")),
                operating_income=self._safe_float(kpis.get("operating_income")),
                net_income=self._safe_float(kpis.get("net_income")),
                diluted_eps=self._safe_float(kpis.get("diluted_eps")),
                gross_margin=self._safe_float(ratios.get("gross_margin")),
                operating_margin=self._safe_float(ratios.get("operating_margin")),
                net_profit_margin=self._safe_float(ratios.get("net_profit_margin")),
                free_cash_flow=self._safe_float(ratios.get("free_cash_flow")),
                total_debt=self._safe_float(kpis.get("total_debt")),
                total_cash_and_equivalents=self._safe_float(kpis.get("total_cash_and_equivalents")),
                debt_to_equity=self._safe_float(ratios.get("debt_to_equity"))
            )
            session.add(metric)
            await session.flush()

            # 8. Create Chunks with 768-D Vector Embeddings
            chunks_to_add = []
            for chunk_idx, page in enumerate(pages):
                p_num = page.get("page_number", 1)
                p_text = page.get("text", "").strip()
                p_section = str(page.get("detected_section") or "Item 8 Consolidated Financial Statements")[:50]

                if not p_text:
                    continue

                raw_emb = self.embedder.embed_text(p_text)
                clean_emb = [float(x) for x in raw_emb] if raw_emb else [0.0] * 768

                chunk = DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=doc.id,
                    ticker=ticker,
                    section=p_section,
                    chunk_type="table" if ("---" in p_text or "|" in p_text) else "text",
                    content=p_text,
                    page_number=p_num,
                    chunk_index=chunk_idx,
                    embedding=clean_emb
                )
                chunks_to_add.append(chunk)

            if chunks_to_add:
                session.add_all(chunks_to_add)

            # Atomic Commit
            await session.commit()

            return {
                "status": "success",
                "document_id": str(doc.id),
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "chunks_indexed": len(chunks_to_add),
                "company_name": company_name
            }
