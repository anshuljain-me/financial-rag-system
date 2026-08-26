import json
import urllib.request
from pathlib import Path
from typing import List, Dict, Any

class CompanyDirectorySearch:
    """
    Universal US SEC Directory Search Engine:
    Downloads and caches the official SEC EDGAR Master Company Registry (10,000+ public companies)
    covering S&P 500, NASDAQ, NYSE, and the Russell 3000.
    """

    SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    CACHE_PATH = Path("data/sec_company_tickers.json")
    _cached_index: List[Dict[str, Any]] = []

    @classmethod
    def _load_sec_index(cls) -> List[Dict[str, Any]]:
        """Loads or downloads the master SEC company registry."""
        if cls._cached_index:
            return cls._cached_index

        cls.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

        # 1. If cached locally, load from disk (<10ms)
        if cls.CACHE_PATH.exists():
            try:
                with open(cls.CACHE_PATH, "r", encoding="utf-8") as f:
                    cls._cached_index = json.load(f)
                    return cls._cached_index
            except Exception:
                pass

        # 2. Fetch fresh master directory from SEC EDGAR
        try:
            req = urllib.request.Request(
                cls.SEC_TICKERS_URL,
                headers={"User-Agent": "FinancialRAGStudio research@financialrag.com"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                raw_data = json.loads(response.read().decode("utf-8"))
                
                # Format: raw_data is a dict of {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
                index = []
                for _, item in raw_data.items():
                    index.append({
                        "ticker": item["ticker"].upper(),
                        "name": item["title"],
                        "cik": str(item["cik_str"]).zfill(10)
                    })
                
                cls._cached_index = index
                
                # Save cache to disk for instant future loads
                with open(cls.CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(cls._cached_index, f)
                    
                print(f"📥 Cached {len(cls._cached_index)} US SEC public companies to {cls.CACHE_PATH}")
                return cls._cached_index
        except Exception as e:
            print(f"⚠️ Could not fetch live SEC feed ({e}). Using offline fallback.")
            return cls._offline_fallback()

    @classmethod
    def _offline_fallback(cls) -> List[Dict[str, Any]]:
        return [
            {"ticker": "AAPL", "name": "Apple Inc.", "cik": "0000320193"},
            {"ticker": "MSFT", "name": "Microsoft Corp", "cik": "0000789019"},
            {"ticker": "NVDA", "name": "NVIDIA Corp", "cik": "0001045810"},
            {"ticker": "AMZN", "name": "Amazon.com Inc.", "cik": "0001018724"},
            {"ticker": "GOOGL", "name": "Alphabet Inc.", "cik": "0001652044"},
            {"ticker": "META", "name": "Meta Platforms, Inc.", "cik": "0001326801"},
            {"ticker": "TSLA", "name": "Tesla, Inc.", "cik": "0001318605"},
            {"ticker": "VZ", "name": "Verizon Communications Inc", "cik": "0000732712"},
            {"ticker": "VRSN", "name": "VeriSign Inc", "cik": "0001014473"},
            {"ticker": "VRSK", "name": "Verisk Analytics, Inc.", "cik": "0001442145"}
        ]

    @classmethod
    def search(cls, query: str, limit: int = 12) -> List[Dict[str, str]]:
        """
        Instant substring & prefix search over all 10,000+ US public equities.
        """
        q = query.strip().lower()
        if not q:
            return []

        all_companies = cls._load_sec_index()
        
        prefix_matches = []
        substring_matches = []

        for item in all_companies:
            t = item["ticker"].lower()
            n = item["name"].lower()

            if t.startswith(q) or n.startswith(q):
                prefix_matches.append(item)
            elif q in t or q in n:
                substring_matches.append(item)

            if len(prefix_matches) >= limit:
                break

        results = prefix_matches + [m for m in substring_matches if m not in prefix_matches]
        return results[:limit]
