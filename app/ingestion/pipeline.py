import os
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from sqlalchemy import select, delete, and_
from sqlalchemy.exc import IntegrityError

from app.core.database import AsyncSessionLocal
from app.models.domain import Company, Document, FinancialMetric, DocumentChunk
from app.ingestion.hasher import Hasher
from app.ingestion.parser import PDFParser
from app.extraction.kpi_extractor import FinancialExtractor
from app.rag.embedder import FinancialEmbedder

class IngestionPipeline:
    """
    Enterprise-Grade SEC 10-K Ingestion Pipeline:
    1. Cryptographic SHA-256 Deduplication.
    2. Table-Preserving PDF Parsing.
    3. Single-Pass Financial KPI & Summary Extraction.
    4. Safe Idempotent Upsert into PostgreSQL (pgvector).
    """

    def __init__(self):
        self.parser = PDFParser()
        self.extractor = FinancialExtractor()
        self.embedder = FinancialEmbedder()

    async def process_file(self, file_path: Path, ticker_override: Optional[str] = None) -> Dict[str, Any]:
        file_path = Path(file_path)
        if not file_path.exists():
            return {"status": "error", "error": f"File not found: {file_path}"}

        # 1. Compute SHA-256 Hash
        file_hash = Hasher.compute_sha256(file_path)

        async with AsyncSessionLocal() as session:
            stmt_hash = select(Document).where(Document.file_hash == file_hash)
            res_hash = await session.execute(stmt_hash)
            existing_doc = res_hash.scalars().first()

            if existing_doc:
                return {
                    "status": "cached",
                    "message": f"Filing {file_path.name} already indexed in database.",
                    "document_id": str(existing_doc.id),
                    "ticker": existing_doc.ticker,
                    "fiscal_year": existing_doc.fiscal_year
                }

        # 2. Structure-Aware Parsing
        parsed_doc = self.parser.parse_pdf(file_path)
        combined_text = "\n\n".join([chunk.text for chunk in parsed_doc.chunks])
        if not combined_text.strip():
            combined_text = f"SEC Form 10-K Annual Report for {ticker_override or 'Company'}."

        # 3. Single-Pass KPI & Summary Extraction
        extracted = self.extractor.extract_kpis_and_summary(combined_text)
        kpis = extracted["kpis"]
        ratios = extracted["calculated_ratios"]
        summary = extracted["summary"]

        ticker_val = (ticker_override or kpis.get("ticker") or file_path.stem.split("_")[0]).upper()
        comp_name = kpis.get("company_name") or f"{ticker_val} Inc."
        f_year = int(kpis.get("fiscal_year") or 2025)
        f_period = f"FY{f_year}"

        async with AsyncSessionLocal() as session:
            # 4. Safe Company Upsert / Lookup
            stmt_comp = select(Company).where(Company.ticker == ticker_val)
            res_comp = await session.execute(stmt_comp)
            company = res_comp.scalars().first()

            if not company:
                company = Company(
                    ticker=ticker_val,
                    company_name=comp_name,
                    sector="Equities",
                    industry="Public Corporations"
                )
                session.add(company)
                try:
                    await session.flush()
                except IntegrityError:
                    await session.rollback()
                    stmt_comp_retry = select(Company).where(Company.ticker == ticker_val)
                    res_retry = await session.execute(stmt_comp_retry)
                    company = res_retry.scalars().first()

            # 5. Safe Document Upsert
            stmt_doc_check = select(Document).where(
                and_(
                    Document.company_id == company.id,
                    Document.fiscal_year == f_year,
                    Document.form_type == "10-K"
                )
            )
            res_doc_check = await session.execute(stmt_doc_check)
            target_doc = res_doc_check.scalars().first()

            if target_doc:
                target_doc.file_name = file_path.name
                target_doc.file_hash = file_hash
                target_doc.executive_summary = summary.get("executive_summary", "")
                target_doc.key_risks = summary.get("key_risks", [])
                target_doc.growth_catalysts = summary.get("growth_catalysts", [])
                
                await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == target_doc.id))
                await session.execute(delete(FinancialMetric).where(FinancialMetric.document_id == target_doc.id))
                await session.flush()
            else:
                target_doc = Document(
                    company_id=company.id,
                    ticker=ticker_val,
                    form_type="10-K",
                    fiscal_year=f_year,
                    fiscal_period=f_period,
                    file_name=file_path.name,
                    file_hash=file_hash,
                    executive_summary=summary.get("executive_summary", ""),
                    key_risks=summary.get("key_risks", []),
                    growth_catalysts=summary.get("growth_catalysts", [])
                )
                session.add(target_doc)
                await session.flush()

            # 6. Insert Financial Metrics
            financial_metric = FinancialMetric(
                document_id=target_doc.id,
                company_id=company.id,
                ticker=ticker_val,
                fiscal_year=f_year,
                revenue=kpis.get("revenue"),
                cost_of_revenue=kpis.get("gross_profit") and kpis.get("revenue") and (kpis.get("revenue") - kpis.get("gross_profit")),
                gross_profit=kpis.get("gross_profit"),
                operating_income=kpis.get("operating_income"),
                net_income=kpis.get("net_income"),
                diluted_eps=kpis.get("diluted_eps"),
                free_cash_flow=ratios.get("free_cash_flow"),
                total_debt=kpis.get("total_debt"),
                total_cash_and_equivalents=kpis.get("total_cash_and_equivalents"),
                shareholders_equity=kpis.get("shareholders_equity"),
                gross_margin=ratios.get("gross_margin"),
                operating_margin=ratios.get("operating_margin"),
                net_profit_margin=ratios.get("net_profit_margin"),
                debt_to_equity=ratios.get("debt_to_equity")
            )
            session.add(financial_metric)

            # 7. Generate Embeddings and Insert Chunks
            chunks_to_insert = []
            for c in parsed_doc.chunks:
                emb = self.embedder.embed_text(c.text)
                chunk_obj = DocumentChunk(
                    document_id=target_doc.id,
                    ticker=ticker_val,
                    section=c.section or "ITEM 8",
                    chunk_type=c.chunk_type or "text",
                    content=c.text,
                    page_number=c.page_number or 1,
                    chunk_index=c.chunk_index or 0,
                    embedding=emb
                )
                chunks_to_insert.append(chunk_obj)

            session.add_all(chunks_to_insert)
            await session.commit()

        return {
            "status": "success",
            "document_id": str(target_doc.id),
            "ticker": ticker_val,
            "company_name": comp_name,
            "fiscal_year": f_year,
            "chunks_indexed": len(chunks_to_insert),
            "kpis": kpis,
            "ratios": ratios
        }
