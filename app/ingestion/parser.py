import pymupdf as fitz
import re
from pathlib import Path
from typing import List, Dict, Any

class SECDocumentParser:
    """
    High-Fidelity Structure-Aware SEC Document Parser.
    Extracts text, preserves financial tables as clean Markdown grids, and tags SEC Items.
    """

    ITEM_PATTERNS = {
        "ITEM 1": re.compile(r"item\s+1[.:\s\-]+business", re.IGNORECASE),
        "ITEM 1A": re.compile(r"item\s+1a[.:\s\-]+risk\s+factors", re.IGNORECASE),
        "ITEM 7": re.compile(r"item\s+7[.:\s\-]+management'?s\s+discussion", re.IGNORECASE),
        "ITEM 8": re.compile(r"item\s+8[.:\s\-]+financial\s+statements", re.IGNORECASE)
    }

    def parse_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF filing not found: {file_path}")

        doc = fitz.open(str(file_path))
        parsed_pages = []
        current_section = "GENERAL"

        for page_idx, page in enumerate(doc, 1):
            text = page.get_text("text") or ""
            
            for section_name, pattern in self.ITEM_PATTERNS.items():
                if pattern.search(text):
                    current_section = section_name
                    break

            tables_md = []
            try:
                tabs = page.find_tables()
                if tabs and len(tabs.tables) > 0:
                    for tab in tabs:
                        df_tab = tab.to_pandas()
                        if not df_tab.empty:
                            tables_md.append(df_tab.to_markdown(index=False))
            except Exception:
                pass

            combined_page_content = text
            if tables_md:
                combined_page_content += "\n\n[EXTRACTED FINANCIAL TABLES]:\n" + "\n\n".join(tables_md)

            parsed_pages.append({
                "page_number": page_idx,
                "section": current_section,
                "content": combined_page_content.strip()
            })

        doc.close()
        return parsed_pages

# Aliases for 100% backward compatibility
DocumentParser = SECDocumentParser
PDFParser = SECDocumentParser
