#!/usr/bin/env python3
"""
Script to run the full data collection pipeline.
Downloads papers from ArXiv, extracts PDF text, and chunks into JSONL.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.data.arxiv_collector import ArxivCollector
from src.data.pdf_extractor import PDFExtractor
from src.data.text_chunker import TextChunker

logger = get_logger("scripts.collect_papers")

def main():
    parser = argparse.ArgumentParser(description="Collect and process ArXiv papers.")
    parser.add_argument("--num_papers", type=int, help="Number of papers to download", default=None)
    parser.add_argument("--categories", nargs="+", help="ArXiv categories (e.g. cs.LG cs.AI)", default=None)
    args = parser.parse_args()
    
    cfg = load_config()
    
    # Overrides
    if args.num_papers:
        cfg.data_collection.num_papers = args.num_papers
    if args.categories:
        cfg.data_collection.categories = args.categories
        
    logger.info("=== Starting Data Collection Pipeline ===")
    
    # Step 1: Download
    collector = ArxivCollector(cfg)
    collector.fetch_papers()
    
    # Step 2: Extract Text
    extractor = PDFExtractor(cfg)
    extractor.process_all()
    
    # Step 3: Chunk Text
    chunker = TextChunker(cfg)
    chunker.process_all()
    
    logger.info("=== Data Collection Pipeline Complete ===")

if __name__ == "__main__":
    main()
