import math
import time
from typing import List, Dict, Any, Optional
from sqlalchemy import select, and_
from rank_bm25 import BM25Okapi
from app.core.database import SyncSessionLocal
from app.models.domain import DocumentChunk
from app.rag.embedder import FinancialEmbedder

class HybridRetriever:
    """
    100% Thread-Safe Synchronous Hybrid Search Engine:
    Combines dense pgvector cosine similarity with sparse BM25 keyword search via RRF.
    Zero asyncio event-loop dependency for complete stability on Streamlit Cloud.
    """

    def __init__(self, k_rrf: int = 60):
        self.embedder = FinancialEmbedder()
        self.k_rrf = k_rrf

    def retrieve_sync(
        self,
        query: str,
        ticker: Optional[str] = None,
        section: Optional[str] = None,
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        with SyncSessionLocal() as session:
            stmt = select(DocumentChunk)
            filters = []
            
            if ticker and ticker.upper() not in ["ALL", "PORTFOLIO", ""]:
                filters.append(DocumentChunk.ticker == ticker.upper())
            if section:
                filters.append(DocumentChunk.section == section)

            if filters:
                stmt = stmt.where(and_(*filters))

            stmt = stmt.limit(300)
            chunks = session.execute(stmt).scalars().all()

        if not chunks:
            return []

        # 1. Dense Semantic Retrieval (Vector Cosine Distance)
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

        # 2. Sparse Lexical Retrieval (BM25 Keyword Matching)
        chunk_texts = [getattr(c, "content", getattr(c, "chunk_text", "")) or "" for c in chunks]
        tokenized_corpus = [t.lower().split() if t else [""] for t in chunk_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        sparse_candidates = [(chunks[i], bm25_scores[i]) for i in range(len(chunks))]
        sparse_ranked = [c for c, _ in sorted(sparse_candidates, key=lambda x: x[1], reverse=True)]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        chunk_map = {}

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
            c_text = getattr(chunk, "content", getattr(chunk, "chunk_text", "")) or ""
            results.append({
                "id": str(chunk.id),
                "ticker": chunk.ticker,
                "section": chunk.section or "SEC Disclosures",
                "page_number": chunk.page_number or 1,
                "chunk_text": c_text,
                "content": c_text,
                "rrf_score": rrf_scores[cid]
            })

        return results

    async def retrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        section: Optional[str] = None,
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        return self.retrieve_sync(query, ticker, section, top_k)

FinancialHybridRetriever = HybridRetriever
HybridSearchRetriever = HybridRetriever
