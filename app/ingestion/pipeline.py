import os
import uuid
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models.domain import Company, Document, FinancialMetric, DocumentChunk
from app.ingestion.hasher import FileHasher
from app.ingestion.parser import PDFParser
from app.extraction.kpi_extractor import FinancialExtractor
from app.rag.embedder import FinancialEmbedder

logger = logging.getLogger(__name__)

def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace("$", "").replace(",", "").replace("M", "").strip()
    try:
        return float(s)
    except Exception:
        return None

class IngestionPipeline:
    def __init__(self):
        self.hasher = FileHasher()
        self.parser = PDFParser()
        self.extractor = FinancialExtractor()
        self.embedder = FinancialEmbedder()

    async def process_file(self, pdf_path: Path, ticker_override: Optional[str] = None) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        file_hash = self.hasher.compute_hash(pdf_path)

        # 1. Deduplication Check
        async with AsyncSessionLocal() as session:
            stmt = select(Document).where(Document.file_hash == file_hash)
            res = await session.execute(stmt)
            existing_doc = res.scalar_one_or_none()
            if existing_doc:
                return {
                    "status": "cached",
                    "document_id": str(existing_doc.id),
                    "ticker": existing_doc.ticker,
                    "message": "Filing previously ingested. Loaded from database."
                }

        # 2. Structure-Aware Parsing
        parse_result = self.parser.parse_pdf(pdf_path)
        metadata = parse_result.get("metadata", {})
        ticker = (ticker_override or metadata.get("ticker") or "UNKNOWN").upper()
        chunks = parse_result.get("chunks", [])

        # 3. Single-Pass Extraction
        combined_sample_text = "\n\n".join([c.get("content") or c.get("text") or "" for c in chunks[:15]])
        extracted = self.extractor.extract_kpis_and_summary(combined_sample_text)
        kpis = extracted.get("kpis", {})
        ratios = extracted.get("calculated_ratios", {})
        summary_data = extracted.get("summary", {})

        doc_year = int(kpis.get("fiscal_year") or metadata.get("fiscal_year") or (2025 if "2025" in str(pdf_path) else 2024))
        doc_period = str(kpis.get("fiscal_period") or metadata.get("fiscal_period") or f"FY{doc_year}")

        # 4. Database Transaction
        async with AsyncSessionLocal() as session:
            c_stmt = select(Company).where(Company.ticker == ticker)
            c_res = await session.execute(c_stmt)
            company = c_res.scalar_one_or_none()

            if not company:
                company_name = kpis.get("company_name") or metadata.get("company_name") or f"{ticker} Inc."
                company = Company(
                    id=uuid.uuid4(),
                    ticker=ticker,
                    company_name=company_name,
                    sector="Technology",
                    industry="Public Equities"
                )
                session.add(company)
                await session.flush()

            # Create Document
            doc = Document(
                id=uuid.uuid4(),
                company_id=company.id,
                ticker=ticker,
                fiscal_year=doc_year,
                fiscal_period=doc_period,
                form_type="10-K",
                executive_summary=summary_data.get("executive_summary", ""),
                key_risks=json.dumps(summary_data.get("key_risks", [])) if isinstance(summary_data.get("key_risks"), list) else str(summary_data.get("key_risks", "")),
                growth_catalysts=json.dumps(summary_data.get("growth_catalysts", [])) if isinstance(summary_data.get("growth_catalysts"), list) else str(summary_data.get("growth_catalysts", "")),
                file_hash=file_hash
            )
            session.add(doc)
            await session.flush()

            # Create Financial Metrics
            metric = FinancialMetric(
                id=uuid.uuid4(),
                document_id=doc.id,
                revenue=_to_float(kpis.get("revenue")),
                gross_profit=_to_float(kpis.get("gross_profit")),
                operating_income=_to_float(kpis.get("operating_income")),
                net_income=_to_float(kpis.get("net_income")),
                diluted_eps=_to_float(kpis.get("diluted_eps")),
                free_cash_flow=_to_float(kpis.get("free_cash_flow") or ratios.get("free_cash_flow")),
                operating_cash_flow=_to_float(kpis.get("operating_cash_flow")),
                capital_expenditures=_to_float(kpis.get("capital_expenditures")),
                total_cash_and_equivalents=_to_float(kpis.get("total_cash_and_equivalents")),
                total_debt=_to_float(kpis.get("total_debt")),
                shareholders_equity=_to_float(kpis.get("shareholders_equity")),
                gross_margin=_to_float(ratios.get("gross_margin")),
                operating_margin=_to_float(ratios.get("operating_margin")),
                net_profit_margin=_to_float(ratios.get("net_profit_margin")),
                debt_to_equity=_to_float(ratios.get("debt_to_equity"))
            )
            session.add(metric)

            # 5. Embed and Insert Chunks
            for chunk_idx, chunk in enumerate(chunks):
                c_text = chunk.get("content") or chunk.get("text") or ""
                if not c_text.strip():
                    continue
                c_emb = self.embedder.embed_text(c_text)

                chunk_obj = DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=doc.id,
                    ticker=ticker,
                    section=chunk.get("section") or "SEC Disclosures",
                    chunk_type=chunk.get("chunk_type") or "text",
                    content=c_text,
                    page_number=int(chunk.get("page_number") or 1),
                    chunk_index=chunk_idx,
                    embedding=c_emb
                )
                session.add(chunk_obj)

            await session.commit()

            return {
                "status": "success",
                "document_id": str(doc.id),
                "ticker": ticker,
                "fiscal_year": doc_year,
                "chunks_indexed": len(chunks)
            }
