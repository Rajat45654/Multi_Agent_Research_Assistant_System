"""
Embeddings generation using sentence-transformers.

Provides a unified interface to generate embeddings for text chunks.
"""

import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer

from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)

class Embedder:
    """Generates embeddings using sentence-transformers."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_name = config.embeddings.model_name
        
        # Determine device
        if config.embeddings.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.embeddings.device
            
        logger.info(f"Loading embedding model '{self.model_name}' on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dimension}")

    def encode(self, texts: Union[str, List[str]], show_progress_bar: bool = False):
        """
        Encode a list of texts into embeddings.
        
        Returns:
            numpy.ndarray of shape (len(texts), embedding_dimension)
        """
        return self.model.encode(
            texts,
            batch_size=self.config.embeddings.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True # Crucial for cosine similarity with inner product
        )
