#!/usr/bin/env python3
"""
Script to generate synthetic Q&A pairs using local LLM.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.data.qa_generator import QAGenerator

logger = get_logger("scripts.generate_qa")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Q&A pairs.")
    parser.add_argument("--dry_run", action="store_true", help="Generate dummy data without loading LLM")
    parser.add_argument("--target", type=int, help="Target number of Q&A pairs to generate")
    args = parser.parse_args()
    
    cfg = load_config()
    
    if args.target:
        cfg.qa_generation.target_total = args.target
        
    logger.info("=== Starting Q&A Generation ===")
    if args.dry_run:
        logger.info("DRY RUN MODE ENABLED (No LLM will be loaded)")
        
    generator = QAGenerator(cfg, dry_run=args.dry_run)
    generator.process_all()
    
    logger.info("=== Q&A Generation Complete ===")

if __name__ == "__main__":
    main()
