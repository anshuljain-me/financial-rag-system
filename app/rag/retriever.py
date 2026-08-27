import math
import asyncio
from typing import List, Dict, Any, Optional
from sqlalchemy import select, and_
from rank_bm25 import BM25Okapi
from app.core.database import AsyncSessionLocal
from app.models.domain import DocumentChunk, Document
from app.rag.embedder import FinancialEmbedder

class HybridRetriever:
    """
    Hybrid Search Engine strictly retrieving Annual Form 10-K Disclosures:
    1. Dense Vector Cosine Similarity (pgvector 768-dim embeddings)
    2. Sparse BM25 Keyword Search
    3. Reciprocal Rank Fusion (RRF) for robust multi-modal retrieval.
    """

    def __init__(self, k_rrf: int = 60):
        self.embedder = FinancialEmbedder()
        self.k_rrf = k_rrf

    async def retrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        section: Optional[str] = None,
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            # Strictly filter to 10-K documents to prevent mixing with old quarterly test data
            stmt = select(DocumentChunk, Document).\
                join(Document, DocumentChunk.document_id == Document.id).\
                where(Document.form_type == "10-K")

            filters = []
            if ticker and ticker.upper() not in ["ALL", "PORTFOLIO", ""]:
                filters.append(DocumentChunk.ticker == ticker.upper())
            if section:
                filters.append(DocumentChunk.section == section)

            if filters:
                stmt = stmt.where(and_(*filters))

            stmt = stmt.limit(250)
            res = await session.execute(stmt)
            rows = res.all()

        if not rows:
            return []

        chunks_with_doc = [(chunk, doc) for chunk, doc in rows]
        chunks = [c for c, d in chunks_with_doc]

        # 1. Dense Semantic Retrieval
        query_embedding = self.embedder.embed_query(query)
        dense_candidates = []
        for c in chunks:
            if c.embedding is not None:
                c_emb = list(c.embedding) if hasattr(c.embedding, "__iter__") else c.embedding
                dot_product = sum(a * b for a, b in zip(query_embedding, c_emb))
                norm_q = math.sqrt(sum(a * a for a in query_embedding))
                norm_c = math.sqrt(sum(b * b for b in c_emb))
                sim = dot_product / (norm_q * norm_c) if norm_q and norm_c else 0.0
                dense_candidates.append((c, sim))

        dense_ranked = [c for c, _ in sorted(dense_candidates, key=lambda x: x[1], reverse=True)]

        # 2. Sparse Lexical Retrieval (BM25)
        chunk_texts = [getattr(c, "content", getattr(c, "chunk_text", "")) or "" for c in chunks]
        tokenized_corpus = [t.lower().split() if t else [""] for t in chunk_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        sparse_candidates = [(chunks[i], bm25_scores[i]) for i in range(len(chunks))]
        sparse_ranked = [c for c, _ in sorted(sparse_candidates, key=lambda x: x[1], reverse=True)]

        # 3. Reciprocal Rank Fusion
        rrf_scores = {}
        chunk_map = {}
        doc_map = {c.id: d for c, d in chunks_with_doc}

        for rank, chunk in enumerate(dense_ranked):
            chunk_map[chunk.id] = chunk
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (self.k_rrf + rank + 1))

        for rank, chunk in enumerate(sparse_ranked):
            chunk_map[chunk.id] = chunk
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (self.k_rrf + rank + 1))

        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        results = []
        for cid in sorted_chunk_ids:
            chunk = chunk_map[cid]
            doc = doc_map.get(cid)
            c_text = getattr(chunk, "content", getattr(chunk, "chunk_text", "")) or ""
            fy_str = f"FY{doc.fiscal_year}" if doc and doc.fiscal_year else "10-K"
            results.append({
                "id": str(chunk.id),
                "ticker": chunk.ticker,
                "fiscal_year": doc.fiscal_year if doc else None,
                "fiscal_period": fy_str,
                "section": chunk.section or "Item 8",
                "page_number": chunk.page_number or 1,
                "chunk_text": c_text,
                "content": c_text,
                "rrf_score": rrf_scores[cid]
            })

        return results

FinancialHybridRetriever = HybridRetriever
HybridSearchRetriever = HybridRetriever
