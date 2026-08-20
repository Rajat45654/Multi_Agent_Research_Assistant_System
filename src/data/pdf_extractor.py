"""
PDF Extractor using PyMuPDF (fitz).

Reads downloaded PDFs and extracts text, handling common artifacts.
Saves extracted text to data/raw/papers/{arxiv_id}.txt
"""

import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.config import Config, load_config

logger = get_logger(__name__)

class PDFExtractor:
    """Extracts text from PDF files."""
    
    def __init__(self, config: Config):
        self.config = config
        self.papers_dir = Path(config.paths.papers_dir)

    def extract_text(self, pdf_path: Path) -> str:
        """Extract clean text from a single PDF."""
        text_content = []
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    # Simple text extraction. PyMuPDF handles columns reasonably well in block mode.
                    # We can use get_text("text") or get_text("blocks"). "text" is usually sufficient.
                    text = page.get_text("text")
                    text_content.append(text)
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path.name}: {e}")
            return ""
            
        return "\n\n".join(text_content)

    def process_all(self) -> None:
        """Process all PDFs in the papers directory."""
        pdf_files = list(self.papers_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to extract.")
        
        success_count = 0
        for pdf_path in tqdm(pdf_files, desc="Extracting text"):
            txt_path = pdf_path.with_suffix(".txt")
            if txt_path.exists():
                continue
                
            text = self.extract_text(pdf_path)
            if text.strip():
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                success_count += 1
            else:
                logger.warning(f"No text extracted from {pdf_path.name}")
                
        logger.info(f"Extraction complete. Successfully extracted text from {success_count} new papers.")

if __name__ == "__main__":
    cfg = load_config()
    extractor = PDFExtractor(cfg)
    extractor.process_all()
