import os
import uuid
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy import select, and_, delete

from app.core.database import SyncSessionLocal, sync_engine, Base
from app.models.domain import Company, Document, FinancialMetric, DocumentChunk
from app.ingestion.hasher import compute_sha256
from app.ingestion.parser import parse_pdf_structure
from app.extraction.kpi_extractor import FinancialExtractor
from app.rag.embedder import FinancialEmbedder

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Enterprise-Grade Resilient SEC Form 10-K Ingestion Pipeline.
    100% Thread-Safe, Zero Event-Loop Collisions, Atomic Transactions.
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

    def process_file_sync(self, pdf_path: Path, ticker_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Synchronous, 100% thread-safe ingestion executed via psycopg2.
        Eliminates all asyncio event-loop conflicts in Streamlit threads.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        # Ensure all tables exist
        Base.metadata.create_all(bind=sync_engine)

        file_hash = compute_sha256(pdf_path)

        with SyncSessionLocal() as session:
            try:
                # 1. Parse PDF Structure
                parsed_data = parse_pdf_structure(pdf_path)
                pages = parsed_data.get("pages", [])
                full_text_sample = "\n\n".join([page.get("text", "") for page in pages])

                # 2. Extract Structured KPIs and Executive Summary
                extracted = self.extractor.extract_kpis_and_summary(full_text_sample)
                kpis = extracted.get("kpis", {})
                ratios = extracted.get("calculated_ratios", {})
                summary = extracted.get("summary", {})

                ticker = (ticker_override or kpis.get("ticker") or "UNKNOWN").strip().upper()[:10]
                company_name = str(kpis.get("company_name") or f"{ticker} Inc.")[:255]
                fiscal_year = int(kpis.get("fiscal_year") or 2025)
                fiscal_period = f"FY{fiscal_year}"[:20]

                # 3. Resolve or Create Company
                stmt_comp = select(Company).where(Company.ticker == ticker)
                company = session.execute(stmt_comp).scalar_one_or_none()

                if not company:
                    company = Company(
                        id=uuid.uuid4(),
                        ticker=ticker,
                        company_name=company_name,
                        sector=str(kpis.get("sector", "General"))[:100],
                        industry=str(kpis.get("industry", "Diversified"))[:100]
                    )
                    session.add(company)
                    session.flush()

                # 4. Clean up existing Documents to avoid UNIQUE collisions on file_hash or (company, year, form)
                stmt_old = select(Document.id).where(
                    and_(
                        Document.company_id == company.id,
                        Document.fiscal_year == fiscal_year,
                        Document.form_type == "10-K"
                    )
                )
                old_doc_ids = session.execute(stmt_old).scalars().all()

                stmt_hash = select(Document.id).where(Document.file_hash == file_hash)
                hash_doc_ids = session.execute(stmt_hash).scalars().all()

                all_delete_ids = list(set(old_doc_ids + hash_doc_ids))
                if all_delete_ids:
                    session.execute(delete(FinancialMetric).where(FinancialMetric.document_id.in_(all_delete_ids)))
                    session.execute(delete(DocumentChunk).where(DocumentChunk.document_id.in_(all_delete_ids)))
                    session.execute(delete(Document).where(Document.id.in_(all_delete_ids)))
                    session.flush()

                # 5. Insert Document
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
                session.flush()

                # 6. Insert Financial Metrics
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
                session.flush()

                # 7. Insert Chunks with Vector Embeddings
                chunks_to_add = []
                for chunk_idx, page in enumerate(pages):
                    p_num = page.get("page_number", 1)
                    p_text = page.get("text", "").strip()
                    p_section = str(page.get("detected_section") or "Item 8 Consolidated Financial Statements")[:50]

                    if not p_text:
                        continue

                    raw_emb = self.embedder.embed_text(p_text)
                    clean_emb = [float(x) for x in raw_emb] if (raw_emb and len(raw_emb) == 768) else None

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

                # Commit Transaction
                session.commit()

                return {
                    "status": "success",
                    "document_id": str(doc.id),
                    "ticker": ticker,
                    "fiscal_year": fiscal_year,
                    "chunks_indexed": len(chunks_to_add),
                    "company_name": company_name
                }

            except Exception as e:
                session.rollback()
                logger.exception(f"Error processing file {pdf_path}: {e}")
                raise e

    async def process_file(self, pdf_path: Path, ticker_override: Optional[str] = None) -> Dict[str, Any]:
        """Async compatibility wrapper."""
        return self.process_file_sync(pdf_path, ticker_override)
