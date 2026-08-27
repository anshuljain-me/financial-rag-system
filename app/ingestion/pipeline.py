import os
from pathlib import Path
from typing import Dict, Any, List
from sqlalchemy import select, delete, and_
import json

from app.core.database import AsyncSessionLocal
from app.models.domain import Company, Document, FinancialMetric, DocumentChunk
from app.ingestion.hasher import compute_file_hash
from app.ingestion.parser import SECDocumentParser
from app.extraction.kpi_extractor import FinancialExtractor
from app.rag.embedder import FinancialEmbedder

class IngestionPipeline:
    """
    Enterprise Idempotent Ingestion Pipeline.
    Strictly enforces zero duplicate records across relational and vector databases.
    Business Key: (ticker, fiscal_year, form_type="10-K").
    """

    def __init__(self):
        self.parser = SECDocumentParser()
        self.extractor = FinancialExtractor()
        self.embedder = FinancialEmbedder()

    async def process_file(self, file_path: Path, ticker_override: str = None) -> Dict[str, Any]:
        file_hash = compute_file_hash(file_path)

        # 1. Parse Structure-Aware Document & Tables
        parsed_doc = self.parser.parse_pdf(file_path)
        file_text = "\n\n".join([page["text"] for page in parsed_doc["pages"]])

        # 2. Single-Pass KPI & Summary Extraction
        extracted_data = self.extractor.extract_kpis_and_summary(file_text[:40000])
        kpis = extracted_data["kpis"]
        ratios = extracted_data["calculated_ratios"]
        summary = extracted_data["summary"]

        ticker = (ticker_override or kpis.get("ticker", "UNKNOWN")).upper()
        fiscal_year = kpis.get("fiscal_year", 2025)
        company_name = kpis.get("company_name", f"{ticker} Inc.")
        form_type = "10-K"
        fiscal_period = f"FY{fiscal_year}"

        async with AsyncSessionLocal() as session:
            # 3. Ensure Company Exists
            stmt_comp = select(Company).where(Company.ticker == ticker)
            res_comp = await session.execute(stmt_comp)
            company = res_comp.scalars().first()

            if not company:
                company = Company(ticker=ticker, company_name=company_name)
                session.add(company)
                await session.flush()

            # 4. STRICT IDEMPOTENCY: Delete any existing record for (ticker, fiscal_year, form_type)
            stmt_existing = select(Document).where(
                and_(
                    Document.ticker == ticker,
                    Document.fiscal_year == fiscal_year,
                    Document.form_type == form_type
                )
            )
            res_existing = await session.execute(stmt_existing)
            existing_docs = res_existing.scalars().all()

            if existing_docs:
                existing_doc_ids = [d.id for d in existing_docs]
                # Delete existing vector chunks
                await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id.in_(existing_doc_ids)))
                # Delete existing metrics
                await session.execute(delete(FinancialMetric).where(FinancialMetric.document_id.in_(existing_doc_ids)))
                # Delete existing documents
                await session.execute(delete(Document).where(Document.id.in_(existing_doc_ids)))
                await session.flush()

            # 5. Insert Fresh Document Record
            doc_record = Document(
                company_id=company.id,
                ticker=ticker,
                form_type=form_type,
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                file_hash=file_hash,
                executive_summary=summary.get("executive_summary", ""),
                key_risks=json.dumps(summary.get("key_risks", [])),
                growth_catalysts=json.dumps(summary.get("growth_catalysts", []))
            )
            session.add(doc_record)
            await session.flush()

            # 6. Insert Financial Metric Record
            metric_record = FinancialMetric(
                document_id=doc_record.id,
                revenue=kpis.get("revenue"),
                gross_profit=kpis.get("gross_profit"),
                operating_income=kpis.get("operating_income"),
                net_income=kpis.get("net_income"),
                diluted_eps=kpis.get("diluted_eps"),
                total_cash_and_equivalents=kpis.get("total_cash_and_equivalents"),
                total_debt=kpis.get("total_debt"),
                gross_margin=ratios.get("gross_margin"),
                operating_margin=ratios.get("operating_margin"),
                net_profit_margin=ratios.get("net_profit_margin"),
                debt_to_equity=ratios.get("debt_to_equity"),
                free_cash_flow=ratios.get("free_cash_flow")
            )
            session.add(metric_record)

            # 7. Generate Chunks & Vector Embeddings
            chunk_records = []
            for page in parsed_doc["pages"]:
                page_num = page["page_number"]
                page_text = page["text"]
                sec_section = page.get("section", "ITEM 8")

                paragraphs = [p.strip() for p in page_text.split("\n\n") if len(p.strip()) > 30]
                for p_idx, p_text in enumerate(paragraphs):
                    emb = self.embedder.embed_text(p_text)
                    chunk = DocumentChunk(
                        document_id=doc_record.id,
                        ticker=ticker,
                        section=sec_section,
                        chunk_type="text",
                        content=p_text,
                        page_number=page_num,
                        chunk_index=p_idx,
                        embedding=emb
                    )
                    chunk_records.append(chunk)

            if chunk_records:
                session.add_all(chunk_records)

            await session.commit()

            return {
                "status": "upserted",
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "form_type": form_type,
                "chunks_indexed": len(chunk_records)
            }
