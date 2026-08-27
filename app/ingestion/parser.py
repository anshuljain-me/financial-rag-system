import re
from pathlib import Path
from typing import Dict, Any, List
import pymupdf as fitz
import pdfplumber

class StructureAwarePDFParser:
    """
    Structure-aware parser that extracts textual sections and transforms
    financial tables into Markdown format while detecting SEC items.
    """

    ITEM_PATTERNS = {
        "ITEM 1. BUSINESS": "ITEM 1",
        "ITEM 1A. RISK FACTORS": "ITEM 1A",
        "ITEM 7. MANAGEMENT'S DISCUSSION": "ITEM 7",
        "ITEM 8. CONSOLIDATED FINANCIAL STATEMENTS": "ITEM 8",
        "ITEM 8. FINANCIAL STATEMENTS": "ITEM 8",
        "CONSOLIDATED STATEMENTS OF OPERATIONS": "ITEM 8",
        "CONSOLIDATED BALANCE SHEETS": "ITEM 8",
        "CONSOLIDATED STATEMENTS OF CASH FLOWS": "ITEM 8"
    }

    def _detect_section(self, text: str, current_section: str) -> str:
        text_upper = text.upper()
        for pattern, section_tag in self.ITEM_PATTERNS.items():
            if pattern in text_upper:
                return section_tag
        return current_section

    def _table_to_markdown(self, table: List[List[Any]]) -> str:
        if not table or len(table) < 2:
            return ""
        clean_table = [[str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row] for row in table]
        headers = clean_table[0]
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in clean_table[1:] if any(row)]
        return "\n".join([header_line, sep_line] + rows)

    def parse_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        chunks = []
        doc = fitz.open(str(pdf_path))
        current_section = "GENERAL"
        chunk_idx = 0

        tables_by_page = {}
        try:
            with pdfplumber.open(str(pdf_path)) as plumber_pdf:
                for page_idx, p_page in enumerate(plumber_pdf.pages):
                    extracted_tables = p_page.extract_tables()
                    if extracted_tables:
                        tables_by_page[page_idx + 1] = extracted_tables
        except Exception:
            pass

        for page_num in range(1, len(doc) + 1):
            page = doc[page_num - 1]
            text = page.get_text("text").strip()
            
            current_section = self._detect_section(text, current_section)

            if page_num in tables_by_page:
                for tbl in tables_by_page[page_num]:
                    md_table = self._table_to_markdown(tbl)
                    if md_table:
                        chunks.append({
                            "chunk_index": chunk_idx,
                            "page": page_num,
                            "section": current_section,
                            "type": "table",
                            "text": f"[{current_section} - Financial Statement Table (Page {page_num})]\n{md_table}"
                        })
                        chunk_idx += 1

            if text:
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
                if not paragraphs:
                    paragraphs = [text]

                for p in paragraphs:
                    chunks.append({
                        "chunk_index": chunk_idx,
                        "page": page_num,
                        "section": current_section,
                        "type": "text",
                        "text": f"[{current_section} (Page {page_num})]\n{p}"
                    })
                    chunk_idx += 1

        doc.close()
        return {
            "file_name": Path(pdf_path).name,
            "total_pages": len(doc),
            "chunks": chunks
        }
