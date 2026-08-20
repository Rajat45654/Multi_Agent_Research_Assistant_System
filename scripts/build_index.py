#!/usr/bin/env python3
"""
Script to build FAISS and BM25 indices from processed chunks.
"""

import sys
from pathlib import Path

# Add project root to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.retrieval.embeddings import Embedder
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.bm25_index import BM25Index

logger = get_logger("scripts.build_index")

def main():
    cfg = load_config()
    chunks_file = Path(cfg.paths.processed_dir) / "chunks.jsonl"
    
    if not chunks_file.exists():
        logger.error(f"Chunks file not found at {chunks_file}. Run collect_papers.py first.")
        sys.exit(1)
        
    logger.info("=== Starting Index Building ===")
    
    # Initialize embedder once to save memory
    embedder = Embedder(cfg)
    
    # 1. Build FAISS
    logger.info("Building FAISS index...")
    faiss_idx = FaissIndex(cfg, embedder=embedder)
    faiss_idx.build(chunks_file)
    
    # 2. Build BM25
    logger.info("Building BM25 index...")
    bm25_idx = BM25Index(cfg)
    bm25_idx.build(chunks_file)
    
    logger.info("=== Index Building Complete ===")

if __name__ == "__main__":
    main()
