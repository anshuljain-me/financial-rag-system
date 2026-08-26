import re
import fitz  # PyMuPDF
import pdfplumber
from typing import List, Dict, Any
from pathlib import Path

class FinancialPDFParser:
    """
    Structure-aware parser for SEC financial filings and corporate reports.
    Extracts text, preserves multi-column tabular data as Markdown,
    and identifies SEC Item section headers.
    """

    SEC_SECTION_PATTERNS = [
        (r"(?i)item\s+1\b[.:\s]+business", "Item 1. Business"),
        (r"(?i)item\s+1a\b[.:\s]+risk\s+factors", "Item 1A. Risk Factors"),
        (r"(?i)item\s+7\b[.:\s]+management['’]?s\s+discussion", "Item 7. MD&A"),
        (r"(?i)item\s+8\b[.:\s]+financial\s+statements", "Item 8. Financial Statements"),
        (r"(?i)item\s+9a\b[.:\s]+controls", "Item 9A. Controls and Procedures"),
        (r"(?i)consolidated\s+balance\s+sheets?", "Consolidated Balance Sheets"),
        (r"(?i)consolidated\s+statements?\s+of\s+(?:operations|income)", "Consolidated Statements of Income"),
        (r"(?i)consolidated\s+statements?\s+of\s+cash\s+flows?", "Consolidated Statements of Cash Flows"),
    ]

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def _detect_section(self, text: str, current_section: str) -> str:
        """Detects if a new SEC Item section header begins on this page."""
        for pattern, section_name in self.SEC_SECTION_PATTERNS:
            if re.search(pattern, text):
                return section_name
        return current_section

    def _table_to_markdown(self, table: List[List[Any]]) -> str:
        """Converts extracted raw table grid to clean Markdown format."""
        if not table or len(table) < 2:
            return ""
        
        # Clean empty rows and whitespace
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
            if any(cleaned_row):
                cleaned_table.append(cleaned_row)
        
        if len(cleaned_table) < 2:
            return ""

        headers = cleaned_table[0]
        # Replace empty header names
        headers = [h if h else f"Col_{i+1}" for i, h in enumerate(headers)]
        
        md_lines = []
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for row in cleaned_table[1:]:
            # Pad row if columns don't match header length
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))
            elif len(row) > len(headers):
                row = row[:len(headers)]
            md_lines.append("| " + " | ".join(row) + " |")
            
        return "\n".join(md_lines)

    def parse(self) -> Dict[str, Any]:
        """
        Extracts structured pages, tables, and section hierarchies.
        """
        doc = fitz.open(self.file_path)
        total_pages = len(doc)
        pages_data = []
        current_section = "General Information"

        # Open with pdfplumber for table extraction
        with pdfplumber.open(self.file_path) as plumber_pdf:
            for page_idx in range(total_pages):
                page_num = page_idx + 1
                fitz_page = doc[page_idx]
                page_text = fitz_page.get_text("text")

                # Detect Section Change
                current_section = self._detect_section(page_text, current_section)

                # Extract Tables via pdfplumber
                plumber_page = plumber_pdf.pages[page_idx]
                tables = plumber_page.extract_tables()
                
                md_tables = []
                for tbl in tables:
                    md = self._table_to_markdown(tbl)
                    if md:
                        md_tables.append(md)

                pages_data.append({
                    "page_number": page_num,
                    "section": current_section,
                    "text": page_text,
                    "tables_markdown": md_tables
                })

        doc.close()

        return {
            "file_name": self.file_path.name,
            "total_pages": total_pages,
            "pages": pages_data
        }