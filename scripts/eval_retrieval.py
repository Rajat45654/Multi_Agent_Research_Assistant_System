#!/usr/bin/env python3
"""
Script to evaluate baseline hybrid retrieval performance on generated Q&A pairs.
"""

import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.retrieval.hybrid_retrieval import HybridRetriever

logger = get_logger("scripts.eval_retrieval")

def calculate_mrr(ranks: list) -> float:
    """Mean Reciprocal Rank"""
    if not ranks:
        return 0.0
    return np.mean([1.0 / r if r > 0 else 0.0 for r in ranks])

def calculate_recall(ranks: list, k: int) -> float:
    """Recall at K (Is the correct doc in top K?)"""
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if 0 < r <= k)
    return hits / len(ranks)

def main():
    cfg = load_config()
    qa_file = Path(cfg.paths.processed_dir) / "qa_pairs.json"
    
    if not qa_file.exists():
        logger.error(f"Q&A file not found at {qa_file}. Run generate_qa.py first.")
        sys.exit(1)
        
    logger.info("Loading Q&A pairs...")
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
        
    logger.info("Loading Hybrid Retriever...")
    retriever = HybridRetriever(cfg)
    
    ranks = []
    
    logger.info(f"Evaluating {len(qa_pairs)} queries...")
    for pair in tqdm(qa_pairs, desc="Evaluating Retrieval"):
        query = pair['question']
        target_doc_id = pair['source_arxiv_id']
        
        # Retrieve top 10
        results = retriever.retrieve(query, top_k=10)
        
        # Find rank of target doc
        rank = 0
        for i, res in enumerate(results):
            if res['arxiv_id'] == target_doc_id:
                rank = i + 1
                break
                
        ranks.append(rank)
        
    mrr_10 = calculate_mrr(ranks)
    recall_5 = calculate_recall(ranks, 5)
    recall_10 = calculate_recall(ranks, 10)
    
    print("\n" + "="*40)
    print("      BASELINE RETRIEVAL METRICS      ")
    print("="*40)
    print(f"Total Queries Evaluated : {len(qa_pairs)}")
    print(f"MRR@10                  : {mrr_10:.4f} (Target: >= 0.70)")
    print(f"Recall@5                : {recall_5:.4f}")
    print(f"Recall@10               : {recall_10:.4f}")
    print("="*40 + "\n")
    
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()
