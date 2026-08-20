"""
FAISS Index for semantic search.

Builds and queries a FAISS index over chunk embeddings.
"""

import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.config import Config
from src.retrieval.embeddings import Embedder

logger = get_logger(__name__)

class FaissIndex:
    """Manages FAISS vector index for semantic search."""
    
    def __init__(self, config: Config, embedder: Embedder = None):
        self.config = config
        self.embedder = embedder
        
        self.embeddings_dir = Path(config.paths.embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.embeddings_dir / "faiss.index"
        self.map_file = self.embeddings_dir / "chunk_map.json"
        
        self.index = None
        self.chunk_map = {} # Maps FAISS internal integer ID to chunk metadata
        
    def build(self, chunks_file: Path) -> None:
        """Build FAISS index from chunks."""
        if not self.embedder:
            logger.error("Embedder not provided. Cannot build index.")
            return
            
        logger.info(f"Building FAISS index from {chunks_file}")
        
        chunks = []
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line))
                
        if not chunks:
            logger.error("No chunks found to index.")
            return
            
        texts = [chunk['text'] for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # We normalized embeddings, so Inner Product is equivalent to Cosine Similarity
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        # Build mapping
        self.chunk_map = {
            str(i): {
                "chunk_id": chunk["chunk_id"],
                "arxiv_id": chunk["arxiv_id"],
                "text": chunk["text"]
            }
            for i, chunk in enumerate(chunks)
        }
        
        self.save()
        logger.info(f"FAISS index built with {self.index.ntotal} vectors.")

    def save(self) -> None:
        """Save index and mapping to disk."""
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_file))
        with open(self.map_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_map, f)
            
    def load(self) -> bool:
        """Load index and mapping from disk."""
        if not self.index_file.exists() or not self.map_file.exists():
            return False
            
        self.index = faiss.read_index(str(self.index_file))
        with open(self.map_file, 'r', encoding='utf-8') as f:
            self.chunk_map = json.load(f)
            
        return True

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search the index for the given query."""
        if self.index is None:
            if not self.load():
                logger.error("FAISS index not loaded or found.")
                return []
                
        if not self.embedder:
            logger.error("Embedder not provided. Cannot search.")
            return []
            
        k = top_k or self.config.retrieval.top_k_semantic
        
        query_embedding = self.embedder.encode([query])
        
        # D = distances (scores), I = indices
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for j, idx in enumerate(indices[0]):
            if idx == -1: # FAISS returns -1 if not enough results
                continue
            str_idx = str(idx)
            if str_idx in self.chunk_map:
                res = self.chunk_map[str_idx].copy()
                res["score"] = float(scores[0][j])
                results.append(res)
                
        return results
