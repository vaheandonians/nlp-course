from pathlib import Path
from typing import Optional, List

import PyPDF2

from config import ConfigManager
from utils.validator import validate_args


class Pdf:
    PAGE_SEPARATOR = "\n\n---\n\n"

    @validate_args({"pdf_path": Path})
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.text: str = ""
        self.pages: List[str] = []
        self._validate_pdf()

    def _validate_pdf(self) -> None:
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {self.pdf_path}")
        if self.pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"File {self.pdf_path} is not a PDF")

    def extract_text(self, pages: Optional[List[int]] = None) -> str:
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                pages_to_process = [p-1 for p in pages if 0 < p <= total_pages] if pages else range(total_pages)
                self.pages = [""] * total_pages
                for page_num in pages_to_process:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    self.pages[page_num] = page_text
                self.text = Pdf.PAGE_SEPARATOR.join(self.pages)
                return self.text.strip()
        except Exception as e:
            raise Exception(f"Unexpected error while processing PDF: {str(e)}")

    def get_page_text(self, page_num: int) -> str:
        if not 0 <= page_num - 1 < len(self.pages) or not self.pages[page_num - 1]:
            raise ValueError(f"Page {page_num} not found. Either invalid page number or page not yet extracted")
        return self.pages[page_num - 1]

    def save_text(self, output_path: Path) -> None:
        if not self.text:
            raise ValueError("No text has been extracted yet. Call extract_text() first")
        output_path.write_text(self.text, encoding='utf-8')

if __name__ == "__main__":
    try:
        pdf = Pdf(ConfigManager().pdf_file_path)
        text = pdf.extract_text()
    except Exception as e:
        print(f"Error: {str(e)}")