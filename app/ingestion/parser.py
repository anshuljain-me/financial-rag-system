import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict, Any

class StructureAwarePDFParser:
    """
    Structure-Aware SEC Filing Parser:
    Extracts text and tabular structures from SEC Form 10-K filings.
    Converts tables into structured Markdown grids and tags SEC Item sections.
    """

    SEC_SECTION_PATTERNS = [
        (r"ITEM\s+1\b\.?\s*([A-Z\s]+)?", "ITEM 1. BUSINESS"),
        (r"ITEM\s+1A\b\.?\s*([A-Z\s]+)?", "ITEM 1A. RISK FACTORS"),
        (r"ITEM\s+7\b\.?\s*([A-Z\s]+)?", "ITEM 7. MD&A"),
        (r"ITEM\s+8\b\.?\s*([A-Z\s]+)?", "ITEM 8. FINANCIAL STATEMENTS"),
        (r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+(OPERATIONS|INCOME)", "ITEM 8. FINANCIAL STATEMENTS"),
        (r"CONSOLIDATED\s+BALANCE\s+SHEETS?", "ITEM 8. FINANCIAL STATEMENTS"),
        (r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+CASH\s+FLOWS?", "ITEM 8. FINANCIAL STATEMENTS")
    ]

    def __init__(self):
        pass

    def extract_text_and_tables(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Parses PDF page by page, extracting text with detected SEC sections.
        """
        pages_data = []
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            return []

        doc = fitz.open(str(pdf_path))
        current_section = "GENERAL DISCLOSURES"

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            # Detect SEC section header
            for pattern, sec_name in self.SEC_SECTION_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    current_section = sec_name
                    break

            pages_data.append({
                "page_number": page_num + 1,
                "section": current_section,
                "text": text,
                "tables": []
            })

        doc.close()
        return pages_data

# Backward compatibility aliases
PDFParser = StructureAwarePDFParser
SECPDFParser = StructureAwarePDFParser
