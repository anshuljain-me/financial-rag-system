import math
import time
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from app.rag.embedder import FinancialEmbedder

logger = logging.getLogger(__name__)

class SemanticCacheEntry:
    def __init__(self, query: str, ticker: str, embedding: List[float], payload: Dict[str, Any], timestamp: float):
        self.query = query
        self.ticker = ticker
        self.embedding = embedding
        self.payload = payload
        self.timestamp = timestamp

class FinancialSemanticCache:
    """
    High-Speed Production Semantic Vector Caching Layer:
    1. Tier 0 (Exact Match Hash Cache): 0.1ms latency (No Embedding API call needed).
    2. Tier 1 (Semantic Vector Cosine Similarity): 0.86 threshold for financial paraphrases.
    """

    def __init__(self, similarity_threshold: float = 0.86, ttl_seconds: int = 86400):
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.embedder = FinancialEmbedder()
        self._exact_cache: Dict[str, SemanticCacheEntry] = {}
        self._vector_cache: List[SemanticCacheEntry] = []
        self._stats = {"hits": 0, "misses": 0, "exact_hits": 0, "semantic_hits": 0}

    def _clean_key(self, query: str, ticker: str) -> str:
        return f"{(ticker or 'ALL').upper()}:::{query.strip().lower()}"

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        return (dot / (norm_a * norm_b)) if norm_a and norm_b else 0.0

    def get_semantic(self, query: str, ticker: str = "ALL") -> Optional[Tuple[Dict[str, Any], float]]:
        if not query.strip():
            return None

        clean_ticker = (ticker or "ALL").upper()
        now = time.time()
        exact_k = self._clean_key(query, clean_ticker)

        # Tier 0: Instant Exact Match (O(1) Hash Map — 0.1 ms, 0 API calls)
        if exact_k in self._exact_cache:
            entry = self._exact_cache[exact_k]
            if (now - entry.timestamp) < self.ttl:
                self._stats["hits"] += 1
                self._stats["exact_hits"] += 1
                cached_res = dict(entry.payload)
                cached_res["cached"] = True
                cached_res["cache_type"] = "EXACT_HASH_HIT"
                cached_res["cache_similarity"] = 1.0
                cached_res["original_cached_query"] = entry.query
                return cached_res, 1.0
            else:
                del self._exact_cache[exact_k]

        # Evict expired vector entries
        self._vector_cache = [e for e in self._vector_cache if (now - e.timestamp) < self.ttl]

        if not self._vector_cache:
            self._stats["misses"] += 1
            return None

        # Filter candidates by ticker scope
        candidate_entries = [
            e for e in self._vector_cache 
            if e.ticker == clean_ticker or clean_ticker in ["ALL", "PORTFOLIO"] or e.ticker in ["ALL", "PORTFOLIO"]
        ]

        if not candidate_entries:
            self._stats["misses"] += 1
            return None

        # Tier 1: Semantic Vector Matching
        query_emb = self.embedder.embed_query(query)

        best_entry = None
        best_sim = -1.0

        for entry in candidate_entries:
            sim = self._cosine_similarity(query_emb, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry and best_sim >= self.threshold:
            self._stats["hits"] += 1
            self._stats["semantic_hits"] += 1
            cached_res = dict(best_entry.payload)
            cached_res["cached"] = True
            cached_res["cache_type"] = "SEMANTIC_VECTOR_HIT"
            cached_res["cache_similarity"] = round(best_sim, 4)
            cached_res["original_cached_query"] = best_entry.query
            return cached_res, best_sim

        self._stats["misses"] += 1
        return None

    def set_semantic(self, query: str, ticker: str, payload: Dict[str, Any]):
        if not query.strip() or not payload:
            return

        clean_ticker = (ticker or "ALL").upper()
        now = time.time()
        query_emb = self.embedder.embed_query(query)
        
        entry = SemanticCacheEntry(
            query=query.strip(),
            ticker=clean_ticker,
            embedding=query_emb,
            payload=payload,
            timestamp=now
        )
        
        exact_k = self._clean_key(query, clean_ticker)
        self._exact_cache[exact_k] = entry
        self._vector_cache.append(entry)

    def clear(self, ticker: Optional[str] = None):
        if ticker:
            t = ticker.upper()
            self._vector_cache = [e for e in self._vector_cache if e.ticker != t]
            self._exact_cache = {k: v for k, v in self._exact_cache.items() if not k.startswith(f"{t}:::")}
        else:
            self._vector_cache.clear()
            self._exact_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        hit_ratio = (self._stats["hits"] / total * 100) if total > 0 else 0.0
        return {
            "hits": self._stats["hits"],
            "exact_hits": self._stats["exact_hits"],
            "semantic_hits": self._stats["semantic_hits"],
            "misses": self._stats["misses"],
            "total_requests": total,
            "hit_ratio_pct": round(hit_ratio, 1),
            "cached_entries": len(self._vector_cache)
        }

semantic_cache = FinancialSemanticCache()
