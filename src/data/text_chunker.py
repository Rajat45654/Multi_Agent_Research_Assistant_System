"""
Text chunker using tiktoken.

Splits text files into overlapping chunks based on token count.
Saves chunks to data/processed/chunks.jsonl
"""

import json
import tiktoken
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

from src.utils.logger import get_logger
from src.utils.config import Config, load_config

logger = get_logger(__name__)

class TextChunker:
    """Splits documents into sliding window token chunks."""
    
    def __init__(self, config: Config):
        self.config = config
        self.papers_dir = Path(config.paths.papers_dir)
        self.processed_dir = Path(config.paths.processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunk_size = config.chunking.chunk_size
        self.overlap = config.chunking.overlap
        
        try:
            self.tokenizer = tiktoken.get_encoding(config.chunking.tokenizer)
        except Exception as e:
            logger.warning(f"Could not load tokenizer {config.chunking.tokenizer}, falling back to cl100k_base. Error: {e}")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
        self.chunks_file = self.processed_dir / "chunks.jsonl"

    def chunk_text(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        tokens = self.tokenizer.encode(text, disallowed_special=())
        chunks = []
        
        if not tokens:
            return chunks
            
        step = self.chunk_size - self.overlap
        if step <= 0:
            step = self.chunk_size # Fallback if overlap is incorrectly configured
            
        chunk_index = 0
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunk = {
                "chunk_id": f"{doc_id}_{chunk_index}",
                "arxiv_id": doc_id,
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "chunk_index": chunk_index
            }
            chunks.append(chunk)
            chunk_index += 1
            
        return chunks

    def process_all(self) -> None:
        """Process all text files and create chunks."""
        txt_files = list(self.papers_dir.glob("*.txt"))
        logger.info(f"Found {len(txt_files)} text files to chunk.")
        
        total_chunks = 0
        # Overwrite chunks file if starting fresh
        with open(self.chunks_file, "w", encoding="utf-8") as f:
            for txt_path in tqdm(txt_files, desc="Chunking texts"):
                arxiv_id = txt_path.stem
                
                with open(txt_path, "r", encoding="utf-8") as tf:
                    text = tf.read()
                    
                chunks = self.chunk_text(text, arxiv_id)
                for chunk in chunks:
                    f.write(json.dumps(chunk) + "\n")
                
                total_chunks += len(chunks)
                
        logger.info(f"Chunking complete. Created {total_chunks} total chunks from {len(txt_files)} documents.")

if __name__ == "__main__":
    cfg = load_config()
    chunker = TextChunker(cfg)
    chunker.process_all()
