import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types
from sqlalchemy import select
from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.models.domain import Company, Document, FinancialMetric
from app.rag.retriever import HybridRetriever
from app.core.cache import semantic_cache

settings = get_settings()

class FinancialQAService:
    """
    Enterprise Structured + Unstructured Hybrid Financial Reasoning Engine
    with Explicit Fiscal Year Normalization & 100% Type Safety.
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

    def _safe_fmt_currency(self, val_m) -> str:
        if val_m is None:
            return "N/A"
        try:
            v = float(val_m)
            if abs(v) >= 1000:
                return f"${v / 1000:.2f}B"
            return f"${v:,.0f}M"
        except Exception:
            return "N/A"

    def _safe_fmt_pct(self, val) -> str:
        if val is None:
            return "N/A"
        try:
            return f"{float(val):.1f}%"
        except Exception:
            return "N/A"

    async def _fetch_all_companies_structured_metrics(self) -> List[Dict[str, Any]]:
        """Fetches the latest annual metrics for every company in the database."""
        async with AsyncSessionLocal() as session:
            stmt = select(Company, Document, FinancialMetric).\
                join(Document, Company.id == Document.company_id).\
                join(FinancialMetric, Document.id == FinancialMetric.document_id).\
                where(Document.form_type == "10-K").\
                order_by(Company.ticker, Document.fiscal_year.desc())
            res = await session.execute(stmt)
            rows = res.all()

            latest_map = {}
            for comp, doc, met in rows:
                if comp.ticker not in latest_map:
                    latest_map[comp.ticker] = {
                        "ticker": comp.ticker,
                        "company_name": comp.company_name,
                        "fiscal_year": doc.fiscal_year,
                        "fiscal_period": f"FY{doc.fiscal_year}",
                        "revenue_m": met.revenue,
                        "gross_margin_pct": met.gross_margin,
                        "operating_margin_pct": met.operating_margin,
                        "net_margin_pct": met.net_profit_margin,
                        "net_income_m": met.net_income,
                        "diluted_eps": met.diluted_eps,
                        "free_cash_flow_m": met.free_cash_flow,
                        "total_debt_m": met.total_debt,
                        "cash_m": met.total_cash_and_equivalents,
                        "debt_to_equity": met.debt_to_equity
                    }
            return list(latest_map.values())

    async def answer_question(self, question: str, ticker: str = "ALL", top_k: int = 6) -> Dict[str, Any]:
        start_time = time.perf_counter()
        target_ticker = None if ticker.upper() in ["ALL", "PORTFOLIO", ""] else ticker.upper()

        # Step 0: Check Semantic Vector Cache
        cached_result = semantic_cache.get_semantic(query=question, ticker=ticker)
        if cached_result:
            payload, _ = cached_result
            payload["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
            return payload

        all_metrics = await self._fetch_all_companies_structured_metrics()

        # Step 1: Execute Hybrid Search for qualitative context
        retrieved_chunks = await self.retriever.retrieve(query=question, ticker=target_ticker, top_k=top_k)

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
            context_blocks.append(f"--- EXCERPT {idx} [{c_ticker} | Form 10-K | {section} | Page {page_num}] ---\n{chunk_text}")

        # Format clean, null-safe database scorecard
        structured_portfolio_text = "PORTFOLIO FINANCIAL DATABASE (ALL INGESTED ANNUAL 10-Ks):\n"
        for m in all_metrics:
            rev_str = self._safe_fmt_currency(m['revenue_m'])
            net_str = self._safe_fmt_currency(m['net_income_m'])
            fcf_str = self._safe_fmt_currency(m['free_cash_flow_m'])
            gm_str = self._safe_fmt_pct(m['gross_margin_pct'])
            om_str = self._safe_fmt_pct(m['operating_margin_pct'])
            nm_str = self._safe_fmt_pct(m['net_margin_pct'])
            de_str = f"{m['debt_to_equity']:.2f}x" if m['debt_to_equity'] is not None else "N/A"

            structured_portfolio_text += (
                f"* {m['ticker']} ({m['company_name']} - FY{m['fiscal_year']}): "
                f"Revenue={rev_str}, Gross Margin={gm_str}, "
                f"Op. Margin={om_str}, Net Margin={nm_str}, "
                f"Net Income={net_str}, FCF={fcf_str}, Debt/Equity={de_str}\n"
            )

        narrative_context = "\n\n".join(context_blocks)

        # Step 2: Strict Direct Institutional Prompt with Explicit Fiscal Year Tagging
        system_prompt = f"""
You are a Lead Equity Research Analyst and CFA Charterholder.

CRITICAL INSTRUCTIONS:
1. DIRECT ANSWER FIRST: In the very first sentence, state the direct conclusion (name the winning company, its exact metric, and its specific Fiscal Year, e.g. 'Palantir (PLTR - FY2025)').
2. EXPLICIT FISCAL YEAR TRANSPARENCY:
   - Always state the exact Fiscal Year (e.g. FY2025, FY2024) for EVERY company mentioned so the analyst knows which period is being compared.
   - When ranking or comparing, check ALL companies in the Portfolio Financial Database below (Apple, Alphabet, Tesla, Palantir, Microsoft, Verisign, etc.).
   - Present a clean, numbered ranking list showing: Rank. TICKER (Company - FYXXXX): X.X%.
3. CONCISE & INSTITUTIONAL: Do not dump unnecessary raw balance sheet walls of text. Keep it sharp and executive-ready.
4. Conclude with a brief parenthetical citation (e.g., Form 10-K Item 8 / Item 1A).

{structured_portfolio_text}

[NARRATIVE SEC FILING EXCERPTS]:
{narrative_context}

[USER QUESTION]:
{question}

Direct Institutional Response:
"""

        # Step 3: Multi-Model Generation
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
            generated_answer = "Unable to process query at this moment. Please try again."

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        response_payload = {
            "answer": generated_answer,
            "citations": citations_list,
            "ticker": ticker,
            "cached": False,
            "latency_ms": elapsed_ms
        }

        semantic_cache.set_semantic(query=question, ticker=ticker, payload=response_payload)
        return response_payload
