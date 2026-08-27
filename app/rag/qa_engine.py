import json
import time
import asyncio
import re
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types
from sqlalchemy import select, and_
from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.models.domain import Company, Document, FinancialMetric
from app.rag.retriever import HybridRetriever
from app.core.cache import semantic_cache

settings = get_settings()

class FinancialQAService:
    """
    Enterprise Structured + Unstructured Hybrid Financial Reasoning Engine.
    Features:
    1. Greeting / Conversational Intent Router (Direct response for 'how are you' / 'hello').
    2. Explicit Fiscal Year Tagging across all comparative queries.
    3. Hybrid SQL Metrics + Vector Narrative Retrieval.
    """

    GREETING_PATTERN = r"^(hi|hello|hey|how are you|how are u|who are you|what can you do|help|good morning|good evening|greetings)[\s\?\!\.]*$"

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

    def _is_greeting(self, question: str) -> bool:
        return bool(re.match(self.GREETING_PATTERN, question.strip().lower()))

    async def _fetch_all_companies_structured_metrics(self) -> List[Dict[str, Any]]:
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
                        "revenue_m": met.revenue or 0.0,
                        "gross_margin_pct": met.gross_margin or 0.0,
                        "operating_margin_pct": met.operating_margin or 0.0,
                        "net_margin_pct": met.net_profit_margin or 0.0,
                        "net_income_m": met.net_income or 0.0,
                        "diluted_eps": met.diluted_eps or 0.0,
                        "free_cash_flow_m": met.free_cash_flow or 0.0,
                        "total_debt_m": met.total_debt or 0.0,
                        "cash_m": met.total_cash_and_equivalents or 0.0,
                        "debt_to_equity": met.debt_to_equity or 0.0
                    }
            return list(latest_map.values())

    async def answer_question(self, question: str, ticker: str = "ALL", top_k: int = 6) -> Dict[str, Any]:
        start_time = time.perf_counter()
        clean_q = question.strip()

        # 1. Immediate Conversational Greeting Router
        if self._is_greeting(clean_q):
            greeting_msg = (
                "Hello! I am doing well, thank you. I am your **AI Equity Research & Financial Copilot**.\n\n"
                "I can assist you with:\n"
                "* **Financial Comparisons:** Compare Gross/Operating Margins, Revenue Scale, and Free Cash Flow across companies.\n"
                "* **Multi-Year Financial Trends:** Trace revenue growth and margin durability over 10-K filings.\n"
                "* **SEC Item 1A Risks & MD&A:** Analyze business models, operational risks, and regulatory disclosures.\n"
                "* **Balance Sheet & Solvency:** Evaluate Cash vs. Total Debt and Debt-to-Equity leverage.\n\n"
                "How can I assist your equity research today?"
            )
            return {
                "answer": greeting_msg,
                "citations": [],
                "ticker": ticker,
                "cached": False,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2)
            }

        # 2. Check Semantic Vector Cache (Only for real financial questions)
        cached_result = semantic_cache.get_semantic(query=clean_q, ticker=ticker)
        if cached_result:
            payload, _ = cached_result
            payload["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
            return payload

        target_ticker = None if ticker.upper() in ["ALL", "PORTFOLIO", ""] else ticker.upper()
        all_metrics = await self._fetch_all_companies_structured_metrics()

        # 3. Hybrid Search for qualitative context
        retrieved_chunks = await self.retriever.retrieve(query=clean_q, ticker=target_ticker, top_k=top_k)

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

        # Format full database portfolio table with explicit fiscal year annotations
        structured_portfolio_text = "PORTFOLIO FINANCIAL DATABASE (ALL INGESTED COMPANIES - EXACT FISCAL YEARS):\n"
        for m in all_metrics:
            rev_str = f"${m['revenue_m']/1000:.2f}B" if m['revenue_m'] >= 1000 else f"${m['revenue_m']:,.0f}M"
            net_str = f"${m['net_income_m']/1000:.2f}B" if abs(m['net_income_m']) >= 1000 else f"${m['net_income_m']:,.0f}M"
            fcf_str = f"${m['free_cash_flow_m']/1000:.2f}B" if abs(m['free_cash_flow_m']) >= 1000 else f"${m['free_cash_flow_m']:,.0f}M"
            structured_portfolio_text += (
                f"* {m['ticker']} ({m['company_name']} - Fiscal Year {m['fiscal_year']}): "
                f"Revenue={rev_str}, Gross Margin={m['gross_margin_pct']:.1f}%, "
                f"Op. Margin={m['operating_margin_pct']:.1f}%, Net Margin={m['net_margin_pct']:.1f}%, "
                f"Net Income={net_str}, Free Cash Flow={fcf_str}, Debt/Equity={m['debt_to_equity']:.2f}x\n"
            )

        narrative_context = "\n\n".join(context_blocks)

        system_prompt = f"""
You are a Senior Equity Research Analyst & Portfolio Manager.

CRITICAL INSTRUCTIONS:
1. ALWAYS GIVE A DIRECT ANSWER FIRST in sentence #1 naming the company, exact metric, and fiscal year.
2. For comparisons or rankings:
   - Check ALL companies in the Portfolio Financial Database below.
   - MANDATORY: Always explicitly specify the exact Fiscal Year (e.g. 'PLTR (FY2025): 82.4%', 'AAPL (FY2024): 46.2%'). Never omit the fiscal year tag.
   - Clearly state whether the ranking compares the latest reported fiscal year for each company or a specific shared fiscal year.
3. Keep the output sharp, concise, and institutional. No unprompted balance sheet dumps.
4. Conclude with a brief parenthetical citation (e.g. Form 10-K Item 8 / Item 1A).

{structured_portfolio_text}

[NARRATIVE SEC FILING EXCERPTS]:
{narrative_context}

[USER QUESTION]:
{clean_q}

Direct Institutional Response:
"""

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

        # Save to semantic cache
        semantic_cache.set_semantic(query=clean_q, ticker=ticker, payload=response_payload)
        return response_payload
