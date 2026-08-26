import json
import time
import asyncio
from typing import Dict, Any, List
from google import genai
from google.genai import types
from app.core.config import get_settings
from app.rag.retriever import HybridRetriever
from app.core.cache import semantic_cache
from app.models.schemas import CitationSource, RAGChatResponse

settings = get_settings()

class FinancialQAService:
    """
    Production-Grade Context-Grounded Financial Reasoning Service
    with Sub-Millisecond Semantic Vector Caching Layer.
    """

    MODEL_CASCADE = [
        "gemini-flash-lite-latest",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.7-flash"
    ]

    def __init__(self):
        self.retriever = HybridRetriever()
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY.strip().strip("'").strip('"'))

    async def answer_question(self, question: str, ticker: str = "ALL", top_k: int = 6) -> Dict[str, Any]:
        """
        1. Checks Semantic Vector Cache (returns in <2ms on semantic hit).
        2. On cache miss, runs Hybrid Search (Dense Vector + Sparse BM25 via RRF).
        3. Generates fact-grounded response with multi-model failover.
        4. Saves response into Semantic Cache.
        """
        start_time = time.perf_counter()
        target_ticker = None if ticker.upper() in ["ALL", "PORTFOLIO", ""] else ticker.upper()

        # Step 0: Semantic Cache Lookup (<2ms)
        cached_result = semantic_cache.get_semantic(query=question, ticker=ticker)
        if cached_result:
            payload, sim_score = cached_result
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            payload["latency_ms"] = elapsed_ms
            return payload

        # Step 1: Hybrid Search Retrieval
        retrieved_chunks = await self.retriever.retrieve(query=question, ticker=target_ticker, top_k=top_k)

        if not retrieved_chunks:
            return {
                "answer": f"I could not locate any indexed SEC Form 10-K disclosures for '{ticker}'. Please ensure the company's annual filing is ingested.",
                "citations": [],
                "ticker": ticker,
                "cached": False
            }

        # Step 2: Build Structured Citation Context
        citations_list = []
        context_blocks = []

        for idx, item in enumerate(retrieved_chunks, 1):
            c_ticker = item.get("ticker", ticker)
            section = item.get("section", "SEC Disclosures")
            page_num = item.get("page_number", 1)
            chunk_text = item.get("chunk_text", "")
            score = item.get("rrf_score", 0.0)

            citations_list.append({
                "ticker": c_ticker,
                "section": section,
                "page_number": page_num,
                "content_snippet": chunk_text[:350] + ("..." if len(chunk_text) > 350 else ""),
                "full_text": chunk_text,
                "score": round(score, 4)
            })

            header = f"--- EXCERPT {idx} [{c_ticker} | Form 10-K | {section} | Page {page_num}] ---"
            context_blocks.append(f"{header}\n{chunk_text}")

        combined_context = "\n\n".join(context_blocks)

        # Step 3: Strict Fact-Grounded Prompt
        system_prompt = f"""
You are a Lead Financial Analyst and CFA Charterholder providing audit-ready equity research.

CRITICAL INSTRUCTIONS:
1. Answer the user's investment question using ONLY the provided SEC Form 10-K excerpts below.
2. DO NOT hallucinate, assume, or extrapolate numbers not present in the excerpts.
3. Explicitly state exact financial line items ($ Millions or $ Billions), margins (%), EPS, and debt/cash figures exactly as disclosed in the source text.
4. Structure your response with clear bullet points and bold key accounting metrics.
5. Conclude with a brief parenthetical citation referencing the specific SEC Item (e.g., Item 8 Financial Statements, Item 1A Risk Factors) and Page Number.

[OFFICIAL SEC FORM 10-K CONTEXT EXCERPTS]:
{combined_context}

[USER QUESTION]:
{question}

Provide an accurate, grounded, professional financial response:
"""

        # Step 4: Multi-Model Generation Cascade
        generated_answer = None
        for model_id in self.MODEL_CASCADE:
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=model_id,
                        contents=system_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            top_p=0.9
                        )
                    )
                    generated_answer = response.text
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        time.sleep(2.0)
                    else:
                        break
            if generated_answer:
                break

        if not generated_answer:
            generated_answer = f"Based on the retrieved SEC 10-K filing excerpts:\n\n" + retrieved_chunks[0]["chunk_text"][:600]

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        response_payload = {
            "answer": generated_answer,
            "citations": citations_list,
            "ticker": ticker,
            "cached": False,
            "latency_ms": elapsed_ms
        }

        # Step 5: Save to Semantic Cache
        semantic_cache.set_semantic(query=question, ticker=ticker, payload=response_payload)

        return response_payload
