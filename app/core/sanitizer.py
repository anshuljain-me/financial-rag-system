import re
from typing import Tuple

class FinancialQuerySanitizer:
    """
    Input Sanitization & Prompt Injection Defense for Financial Inquiries.
    """

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"disregard\s+(all\s+)?(previous|prior)\s+prompts",
        r"system\s*prompt",
        r"you\s+are\s+now\s+(a|an)?",
        r"output\s+all\s+(keys|passwords|env)",
        r"<script.*?>",
        r"javascript:"
    ]

    @classmethod
    def sanitize_query(cls, query: str, max_length: int = 1000) -> Tuple[str, bool]:
        """
        Sanitizes user input and flags malicious injection attempts.
        Returns (sanitized_query, is_flagged).
        """
        if not query:
            return "", False

        cleaned = query.strip()[:max_length]
        is_flagged = False

        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                is_flagged = True
                cleaned = re.sub(pattern, "[FILTERED]", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"<[^>]*>", "", cleaned)
        return cleaned, is_flagged

    @classmethod
    def validate_ticker(cls, ticker: str) -> str:
        """Validates and standardizes equity ticker symbols."""
        if not ticker or ticker.upper() in ["ALL", "PORTFOLIO"]:
            return "ALL"
        clean_t = re.sub(r"[^A-Za-z0-9\.\-]", "", ticker).upper()
        return clean_t[:10]
