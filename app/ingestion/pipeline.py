import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import AsyncSessionLocal
from app.models.domain import Company, Document, FinancialMetric, DocumentChunk
from app.ingestion.hasher import compute_file_sha256
from app.ingestion.parser import FinancialPDFParser
from app.extraction.kpi_extractor import FinancialExtractor
from app.rag.embedder import GeminiEmbedder

class IngestionPipeline:
    """
    Production-grade Financial Ingestion Pipeline:
    1. Content Deduplication via SHA-256 Hashing.
    2. Table-Aware Financial Statement Parsing.
    3. One-Pass Structured KPI & Summary Extraction.
    4. Section-Aware Vector Chunking & pgvector Indexing.
    """

    def __init__(self):
        self.extractor = FinancialExtractor()
        self.embedder = GeminiEmbedder()

    async def process_file(self, file_path: str | Path, ticker_override: str = None) -> Dict[str, Any]:
        file_path = Path(file_path)
        sha256_hash = compute_file_sha256(file_path)
        
        async with AsyncSessionLocal() as session:
            # 1. Check if document was already ingested (Cost Optimization)
            stmt = select(Document).where(Document.sha256_hash == sha256_hash)
            result = await session.execute(stmt)
            existing_doc = result.scalar_one_or_none()
            
            if existing_doc:
                print(f"⚡ [CACHE HIT] File '{file_path.name}' already ingested. Skipping re-processing.")
                return {
                    "status": "cached",
                    "document_id": existing_doc.id,
                    "ticker": existing_doc.ticker,
                    "message": "File was previously ingested. Metrics and embeddings are active."
                }

            print(f"📄 [PROCESSING] New file detected: {file_path.name}")
            
            # 2. Parse Document Structure & Tables
            parser = FinancialPDFParser(file_path)
            parsed_data = parser.parse()

            # Compile text sample for KPI & Summary extraction (focusing on Items 1, 7, 8)
            combined_sample_text = ""
            for page in parsed_data["pages"][:25]: # First 25 pages usually contain core disclosures
                combined_sample_text += f"\n--- Page {page['page_number']} [{page['section']}] ---\n"
                combined_sample_text += page["text"]
                for tbl in page["tables_markdown"]:
                    combined_sample_text += f"\n[TABLE]\n{tbl}\n[/TABLE]\n"

            # 3. Extract Structured KPIs and AI Summary
            print("🧠 [LLM EXTRACTION] Extracting structured financial statements & executive summary...")
            extracted = self.extractor.extract_kpis_and_summary(combined_sample_text)
            kpis = extracted["kpis"]
            ratios = extracted["calculated_ratios"]
            summary = extracted["summary"]

            ticker = ticker_override or kpis.get("ticker") or file_path.stem.split("_")[0].upper()
            company_name = kpis.get("company_name") or ticker
            fiscal_year = kpis.get("fiscal_year") or 2024
            fiscal_period = kpis.get("fiscal_period") or "FY"

            # 4. Upsert Company
            stmt_comp = select(Company).where(Company.ticker == ticker)
            comp_res = await session.execute(stmt_comp)
            company = comp_res.scalar_one_or_none()
            if not company:
                company = Company(ticker=ticker, company_name=company_name)
                session.add(company)
                await session.flush()

            # 5. Create Document Record
            doc_record = Document(
                company_id=company.id,
                ticker=ticker,
                form_type="10-K",
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                file_name=file_path.name,
                file_path=str(file_path.absolute()),
                sha256_hash=sha256_hash,
                total_pages=parsed_data["total_pages"],
                is_processed=True,
                executive_summary=summary.get("executive_summary"),
                key_risks=json_dumps_safe(summary.get("key_risks")),
                growth_catalysts=json_dumps_safe(summary.get("growth_catalysts"))
            )
            session.add(doc_record)
            await session.flush()

            # 6. Save Structured Financial Metrics
            metrics_record = FinancialMetric(
                company_id=company.id,
                document_id=doc_record.id,
                ticker=ticker,
                fiscal_year=fiscal_year,
                period=fiscal_period,
                revenue=kpis.get("revenue"),
                gross_profit=kpis.get("gross_profit"),
                operating_income=kpis.get("operating_income"),
                net_income=kpis.get("net_income"),
                diluted_eps=kpis.get("diluted_eps"),
                operating_cash_flow=kpis.get("operating_cash_flow"),
                capital_expenditures=kpis.get("capital_expenditures"),
                free_cash_flow=ratios.get("free_cash_flow"),
                total_cash_and_equivalents=kpis.get("total_cash_and_equivalents"),
                total_debt=kpis.get("total_debt"),
                shareholders_equity=kpis.get("shareholders_equity"),
                gross_margin=ratios.get("gross_margin"),
                operating_margin=ratios.get("operating_margin"),
                net_profit_margin=ratios.get("net_profit_margin"),
                debt_to_equity=ratios.get("debt_to_equity")
            )
            session.add(metrics_record)

            # 7. Chunk and Embed for Vector Retrieval
            print("🔢 [EMBEDDINGS] Generating vector embeddings for document chunks...")
            chunk_idx = 0
            chunks_to_insert = []
            
            for page in parsed_data["pages"]:
                # Text chunk
                if page["text"].strip():
                    text_content = f"[{ticker} {fiscal_period} Page {page['page_number']} - {page['section']}]\n{page['text']}"
                    embedding = self.embedder.get_embedding(text_content[:2000])
                    chunks_to_insert.append(
                        DocumentChunk(
                            document_id=doc_record.id,
                            ticker=ticker,
                            section=page["section"],
                            page_number=page["page_number"],
                            chunk_index=chunk_idx,
                            chunk_type="text",
                            content=text_content,
                            embedding=embedding
                        )
                    )
                    chunk_idx += 1

                # Table chunks (preserved as atomic Markdown blocks)
                for tbl in page["tables_markdown"]:
                    tbl_content = f"[{ticker} {fiscal_period} TABLE - Page {page['page_number']} - {page['section']}]\n{tbl}"
                    embedding = self.embedder.get_embedding(tbl_content[:2000])
                    chunks_to_insert.append(
                        DocumentChunk(
                            document_id=doc_record.id,
                            ticker=ticker,
                            section=page["section"],
                            page_number=page["page_number"],
                            chunk_index=chunk_idx,
                            chunk_type="table",
                            content=tbl_content,
                            embedding=embedding
                        )
                    )
                    chunk_idx += 1

            session.add_all(chunks_to_insert)
            await session.commit()
            
            print(f"✅ Ingested {file_path.name}: {len(chunks_to_insert)} chunks indexed, KPIs & summary cached.")
            return {
                "status": "success",
                "document_id": doc_record.id,
                "ticker": ticker,
                "chunks_indexed": len(chunks_to_insert),
                "kpis": kpis,
                "ratios": ratios
            }

    async def process_directory(self, directory_path: str | Path) -> List[Dict[str, Any]]:
        """
        Batch ingests an entire directory of financial PDFs.
        """
        dir_path = Path(directory_path)
        pdf_files = list(dir_path.glob("*.pdf")) + list(dir_path.glob("**/*.pdf"))
        
        print(f"📁 Found {len(pdf_files)} PDF filing(s) in {dir_path}")
        results = []
        for pdf_file in pdf_files:
            res = await self.process_file(pdf_file)
            results.append(res)
        return results

def json_dumps_safe(obj: Any) -> str:
    import json
    if isinstance(obj, (list, dict)):
        return json.dumps(obj)
    return str(obj) if obj is not None else ""