import asyncio
from sqlalchemy import select, delete, and_
from app.core.database import AsyncSessionLocal
from app.models.domain import Company, Document, FinancialMetric, DocumentChunk

async def clean_and_deduplicate_database():
    print("=" * 80)
    print("🧹 STARTING DATABASE CLEANUP & DEDUPLICATION (RELATIONAL & VECTOR DB)")
    print("=" * 80)

    async with AsyncSessionLocal() as session:
        # 1. Purge all Quarterly (10-Q) Documents, Chunks, and Metrics
        print("\n[PHASE 1] Purging all Form 10-Q (Quarterly) records...")
        stmt_q_docs = select(Document).where(Document.form_type != "10-K")
        res_q = await session.execute(stmt_q_docs)
        q_docs = res_q.scalars().all()

        q_doc_ids = [d.id for d in q_docs]
        if q_doc_ids:
            await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id.in_(q_doc_ids)))
            await session.execute(delete(FinancialMetric).where(FinancialMetric.document_id.in_(q_doc_ids)))
            await session.execute(delete(Document).where(Document.id.in_(q_doc_ids)))
            print(f"  ✅ Deleted {len(q_doc_ids)} quarterly 10-Q documents and their vector chunks.")
        else:
            print("  ✅ No quarterly 10-Q records found.")

        # 2. Deduplicate Annual 10-K Filings by (ticker, fiscal_year)
        print("\n[PHASE 2] Deduplicating Annual 10-K records by (ticker, fiscal_year)...")
        stmt_10k = select(Document).where(Document.form_type == "10-K").order_by(Document.ticker, Document.fiscal_year, Document.created_at.desc())
        res_10k = await session.execute(stmt_10k)
        all_10k = res_10k.scalars().all()

        seen_keys = set()
        keep_doc_ids = set()
        duplicate_doc_ids = set()

        for doc in all_10k:
            b_key = f"{doc.ticker.upper()}:::{doc.fiscal_year}"
            if b_key not in seen_keys:
                seen_keys.add(b_key)
                keep_doc_ids.add(doc.id)
            else:
                duplicate_doc_ids.add(doc.id)

        if duplicate_doc_ids:
            dup_list = list(duplicate_doc_ids)
            await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id.in_(dup_list)))
            await session.execute(delete(FinancialMetric).where(FinancialMetric.document_id.in_(dup_list)))
            await session.execute(delete(Document).where(Document.id.in_(dup_list)))
            print(f"  ✅ Deleted {len(dup_list)} duplicate annual 10-K documents and vector chunks.")
        else:
            print("  ✅ Zero duplicates found across annual 10-K records.")

        # 3. Clean up any Orphan Vector Chunks
        print("\n[PHASE 3] Checking for orphan vector chunks...")
        active_doc_ids_stmt = select(Document.id)
        res_active_ids = await session.execute(active_doc_ids_stmt)
        active_ids = set(res_active_ids.scalars().all())

        stmt_all_chunks = select(DocumentChunk.id, DocumentChunk.document_id)
        res_chunks = await session.execute(stmt_all_chunks)
        orphan_chunk_ids = [c_id for c_id, d_id in res_chunks.all() if d_id not in active_ids]

        if orphan_chunk_ids:
            await session.execute(delete(DocumentChunk).where(DocumentChunk.id.in_(orphan_chunk_ids)))
            print(f"  ✅ Cleaned {len(orphan_chunk_ids)} orphan vector chunks.")
        else:
            print("  ✅ Zero orphan vector chunks found.")

        # 4. Clean up any Orphan Companies
        stmt_comps = select(Company.id, Company.ticker)
        res_comps = await session.execute(stmt_comps)
        all_comps = res_comps.all()

        for comp_id, ticker in all_comps:
            stmt_check = select(Document.id).where(Document.company_id == comp_id)
            res_c = await session.execute(stmt_check)
            if not res_c.scalars().first():
                await session.execute(delete(Company).where(Company.id == comp_id))
                print(f"  ✅ Cleaned unused company record: {ticker}")

        await session.commit()

        # 5. Print Clean Inventory
        stmt_final_docs = select(Document.ticker, Document.fiscal_year, Document.form_type).order_by(Document.ticker, Document.fiscal_year.desc())
        res_final = await session.execute(stmt_final_docs)
        clean_records = res_final.all()

        print("\n" + "=" * 80)
        print("🏆 CLEAN DATABASE INVENTORY (EXACTLY 1 RECORD PER COMPANY PER YEAR)")
        print("=" * 80)
        for t, fy, f_type in clean_records:
            print(f"  📌 {t:<6} | {f_type} | FY{fy}")
        print(f"\nTotal Clean Annual Filings: {len(clean_records)}")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(clean_and_deduplicate_database())
