"""
Hybrid Retrieval System.

Combines results from FAISS (semantic) and BM25 (keyword) using Reciprocal Rank Fusion.
"""

from typing import List, Dict, Any

from src.utils.logger import get_logger
from src.utils.config import Config
from src.retrieval.embeddings import Embedder
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.bm25_index import BM25Index

logger = get_logger(__name__)

class HybridRetriever:
    """Combines semantic and keyword search."""
    
    def __init__(self, config: Config):
        self.config = config
        
        self.embedder = Embedder(config)
        self.faiss = FaissIndex(config, embedder=self.embedder)
        self.bm25 = BM25Index(config)
        
        # Try loading indices
        if not self.faiss.load():
            logger.warning("FAISS index not found. Needs to be built.")
        if not self.bm25.load():
            logger.warning("BM25 index not found. Needs to be built.")

    def _reciprocal_rank_fusion(self, results_lists: List[List[Dict[str, Any]]], k=60) -> List[Dict[str, Any]]:
        """
        Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).
        RRF_score = sum(1 / (k + rank)) for each rank in results_lists.
        """
        rrf_scores = {}
        chunk_data = {}
        
        for results in results_lists:
            for rank, res in enumerate(results):
                chunk_id = res['chunk_id']
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = res
                    rrf_scores[chunk_id] = 0.0
                
                rrf_scores[chunk_id] += 1.0 / (k + rank + 1)
                
        # Sort by RRF score
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for chunk_id, score in sorted_items:
            res = chunk_data[chunk_id].copy()
            res['hybrid_score'] = score
            final_results.append(res)
            
        return final_results

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval for a query."""
        k_semantic = self.config.retrieval.top_k_semantic
        k_bm25 = self.config.retrieval.top_k_bm25
        k_final = top_k or self.config.retrieval.top_k_final
        
        logger.debug(f"Executing hybrid search for: '{query}'")
        
        semantic_results = self.faiss.search(query, top_k=k_semantic)
        keyword_results = self.bm25.search(query, top_k=k_bm25)
        
        # Weights could be used instead of simple RRF, but RRF is standard for hybrid
        # We use RRF here to avoid issues with uncalibrated score scales between cosine and BM25
        
        combined_results = self._reciprocal_rank_fusion([semantic_results, keyword_results])
        
        return combined_results[:k_final]

if __name__ == "__main__":
    import argparse
    from src.utils.config import load_config
    
    parser = argparse.ArgumentParser(description="Test hybrid retrieval.")
    parser.add_argument("query", type=str, help="Search query")
    args = parser.parse_args()
    
    cfg = load_config()
    retriever = HybridRetriever(cfg)
    results = retriever.retrieve(args.query)
    
    print(f"\nTop results for: '{args.query}'\n")
    for i, res in enumerate(results):
        print(f"[{i+1}] Doc: {res['arxiv_id']} | Score: {res.get('hybrid_score', 0):.4f}")
        print(f"Snippet: {res['text'][:150]}...\n")
