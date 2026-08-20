"""
BM25 Index for keyword search.

Builds and queries a BM25 index over chunk texts.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)

class BM25Index:
    """Manages BM25 index for keyword search."""
    
    def __init__(self, config: Config):
        self.config = config
        
        self.embeddings_dir = Path(config.paths.embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.embeddings_dir / "bm25.pkl"
        self.data_file = self.embeddings_dir / "bm25_data.pkl"
        
        self.bm25 = None
        self.chunks_data = [] # List of chunk metadata matching BM25 corpus order
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25. Can be improved with NLTK/Spacy."""
        return text.lower().split()

    def build(self, chunks_file: Path) -> None:
        """Build BM25 index from chunks."""
        logger.info(f"Building BM25 index from {chunks_file}")
        
        chunks = []
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line))
                
        if not chunks:
            logger.error("No chunks found to index.")
            return
            
        self.chunks_data = [
            {
                "chunk_id": chunk["chunk_id"],
                "arxiv_id": chunk["arxiv_id"],
                "text": chunk["text"]
            }
            for chunk in chunks
        ]
        
        logger.info(f"Tokenizing {len(chunks)} chunks for BM25...")
        corpus_tokens = [self._tokenize(chunk['text']) for chunk in chunks]
        
        logger.info("Fitting BM25 model...")
        self.bm25 = BM25Okapi(corpus_tokens)
        
        self.save()
        logger.info(f"BM25 index built with {len(chunks)} documents.")

    def save(self) -> None:
        """Save index and data to disk."""
        if self.bm25 is None:
            return
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.bm25, f)
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.chunks_data, f)
            
    def load(self) -> bool:
        """Load index and data from disk."""
        if not self.index_file.exists() or not self.data_file.exists():
            return False
            
        with open(self.index_file, 'rb') as f:
            self.bm25 = pickle.load(f)
        with open(self.data_file, 'rb') as f:
            self.chunks_data = pickle.load(f)
            
        return True

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search the index for the given query."""
        if self.bm25 is None:
            if not self.load():
                logger.error("BM25 index not loaded or found.")
                return []
                
        k = top_k or self.config.retrieval.top_k_bm25
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_n_indices:
            if scores[idx] > 0: # Only include if there's some overlap
                res = self.chunks_data[idx].copy()
                res["score"] = float(scores[idx])
                results.append(res)
                
        return results
