import json
import time
import asyncio
from typing import Dict, Any, List, Optional
import yfinance as yf
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
    Unified Institutional Equity Intelligence Engine:
    Combines:
    1. Static SEC Form 10-K Fundamentals (Revenue, Margins, Net Income, EPS, Debt, Cash).
    2. Real-Time Market Valuation (Stock Price, Market Cap, Trailing P/E via yfinance).
    3. Qualitative Hybrid Vector Retrieval (Item 1A Risks, Item 1 Business Strategy).
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
        self._market_data_cache = {}

    def _fetch_live_market_data(self, ticker: str) -> Dict[str, Any]:
        """Fetches live stock price, market cap, and P/E valuation metrics."""
        t = ticker.upper()
        now = time.time()
        
        if t in self._market_data_cache and (now - self._market_data_cache[t]["timestamp"]) < 300:
            return self._market_data_cache[t]["data"]

        try:
            stock = yf.Ticker(t)
            info = stock.fast_info if hasattr(stock, "fast_info") else {}
            
            price = getattr(info, "last_price", None) or getattr(info, "previous_close", None) or 0.0
            mkt_cap = getattr(info, "market_cap", None) or 0.0
            
            full_info = getattr(stock, "info", {}) or {}
            pe_ratio = full_info.get("trailingPE") or full_info.get("forwardPE")
            
            data = {
                "current_price": round(float(price), 2) if price else None,
                "market_cap_b": round(float(mkt_cap) / 1e9, 2) if mkt_cap else None,
                "pe_ratio": round(float(pe_ratio), 2) if pe_ratio else None
            }
        except Exception:
            data = {"current_price": None, "market_cap_b": None, "pe_ratio": None}

        self._market_data_cache[t] = {"data": data, "timestamp": now}
        return data

    async def _fetch_all_companies_comprehensive_data(self) -> List[Dict[str, Any]]:
        """Fetches fundamentals from database and fuses live market valuation."""
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
                    market_info = self._fetch_live_market_data(comp.ticker)
                    price = market_info.get("current_price")
                    eps = met.diluted_eps or 0.0
                    calc_pe = round(price / eps, 2) if price and eps and eps > 0 else market_info.get("pe_ratio")

                    latest_map[comp.ticker] = {
                        "ticker": comp.ticker,
                        "company_name": comp.company_name,
                        "fiscal_year": doc.fiscal_year,
                        "period_label": f"FY{doc.fiscal_year}",
                        "revenue_m": met.revenue or 0.0,
                        "gross_margin_pct": met.gross_margin or 0.0,
                        "operating_margin_pct": met.operating_margin or 0.0,
                        "net_margin_pct": met.net_profit_margin or 0.0,
                        "net_income_m": met.net_income or 0.0,
                        "diluted_eps": eps,
                        "free_cash_flow_m": met.free_cash_flow or 0.0,
                        "total_debt_m": met.total_debt or 0.0,
                        "cash_m": met.total_cash_and_equivalents or 0.0,
                        "debt_to_equity": met.debt_to_equity or 0.0,
                        "market_price": price,
                        "market_cap_b": market_info.get("market_cap_b"),
                        "pe_ratio": calc_pe
                    }
            return list(latest_map.values())

    async def answer_question(self, question: str, ticker: str = "ALL", top_k: int = 6) -> Dict[str, Any]:
        start_time = time.perf_counter()
        target_ticker = None if ticker.upper() in ["ALL", "PORTFOLIO", ""] else ticker.upper()

        # Step 0: Semantic Cache Check
        cached_result = semantic_cache.get_semantic(query=question, ticker=ticker)
        if cached_result:
            payload, _ = cached_result
            payload["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
            return payload

        all_companies = await self._fetch_all_companies_comprehensive_data()

        # Step 1: Qualitative Hybrid Search
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

        # Step 2: Build Complete Portfolio & Valuation Scorecard
        scorecard_text = "PORTFOLIO FINANCIAL & MARKET VALUATION DATABASE (ALL INGESTED COMPANIES):\n"
        for c in all_companies:
            rev_str = f"${c['revenue_m']/1000:.2f}B" if c['revenue_m'] >= 1000 else f"${c['revenue_m']:,.0f}M"
            net_str = f"${c['net_income_m']/1000:.2f}B" if abs(c['net_income_m']) >= 1000 else f"${c['net_income_m']:,.0f}M"
            price_str = f"${c['market_price']}" if c['market_price'] else "N/A"
            pe_str = f"{c['pe_ratio']:.1f}x" if c['pe_ratio'] else "N/A"
            mcap_str = f"${c['market_cap_b']:.2f}B" if c['market_cap_b'] else "N/A"

            scorecard_text += (
                f"* {c['ticker']} ({c['company_name']} - {c['period_label']}): "
                f"Market Price={price_str}, Trailing P/E={pe_str}, Market Cap={mcap_str}, "
                f"Revenue={rev_str}, Gross Margin={c['gross_margin_pct']:.1f}%, "
                f"Op. Margin={c['operating_margin_pct']:.1f}%, Net Margin={c['net_margin_pct']:.1f}%, "
                f"EPS=${c['diluted_eps']:.2f}, FCF=${c['free_cash_flow_m']:,.0f}M, Debt/Equity={c['debt_to_equity']:.2f}x\n"
            )

        narrative_context = "\n\n".join(context_blocks)

        # Step 3: Direct Institutional System Prompt
        system_prompt = f"""
You are a Senior Equity Research Analyst & CFA Charterholder.

CRITICAL INSTRUCTIONS:
1. ALWAYS GIVE A DIRECT ANSWER FIRST:
   - In the very first sentence, state the direct conclusion and exact numbers.
2. EXPLICIT FISCAL PERIOD & VALUATION ANCHORING:
   - When listing companies, ALWAYS include their exact fiscal year (e.g., 'AAPL (FY2024)', 'PLTR (FY2025)', 'TSLA (FY2025)').
   - For valuation queries (P/E Ratio, Market Price, Market Capitalization), use the live market data provided in the scorecard.
3. FOR RANKINGS / COMPARISONS:
   - Analyze ALL ingested companies in the Portfolio Database (Apple, Alphabet, Tesla, Palantir, Microsoft, etc.).
   - Provide a clean, ordered list or compact table with exact figures.
4. Keep the response sharp, quantitative, and concise without dumping unrequested balance sheet tables.

{scorecard_text}

[NARRATIVE SEC FORM 10-K EXCERPTS]:
{narrative_context}

[USER QUESTION]:
{question}

Direct Institutional Response:
"""

        # Step 4: Multi-Model Generation
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
            generated_answer = "Unable to process query at this time. Please retry."

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
