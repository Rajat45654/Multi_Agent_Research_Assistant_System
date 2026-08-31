"""
Retrieval Tool.

A simple Python wrapper that exposes the HybridRetriever as a callable "tool"
that the AgentExecutor can invoke. This keeps the orchestration layer clean
and separates retrieval logic from agent logic.
"""

from typing import List, Dict, Any
from src.retrieval.hybrid_retrieval import HybridRetriever
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalTool:
    """
    Wraps the HybridRetriever as a named tool for the orchestration layer.
    
    Usage:
        tool = RetrievalTool(cfg)
        docs = tool.retrieve("What is attention mechanism?", top_k=10)
    """

    name = "retrieve_documents"
    description = (
        "Searches the indexed research paper database using hybrid semantic + keyword search. "
        "Returns the most relevant document chunks for a given query."
    )

    def __init__(self, cfg: Config):
        self.cfg = cfg
        logger.info("Initializing RetrievalTool (loading FAISS + BM25 indices)...")
        self.retriever = HybridRetriever(cfg)
        logger.info("RetrievalTool ready.")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Runs hybrid retrieval and returns a list of document chunks.
        
        Args:
            query: The natural language search query.
            top_k: Number of chunks to return.
            
        Returns:
            A list of dicts, each with: chunk_id, arxiv_id, text, score
        """
        logger.info(f"RetrievalTool: searching for '{query[:60]}...'")
        results = self.retriever.retrieve(query, top_k=top_k)
        logger.info(f"RetrievalTool: returned {len(results)} chunks.")
        return results

    def get_paper_metadata(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Returns basic metadata for a given ArXiv paper ID.
        Reads from the saved metadata JSON generated during Phase 1.
        """
        import json
        from pathlib import Path

        metadata_file = Path(self.cfg.paths.processed_dir) / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                all_meta = json.load(f)
            return all_meta.get(arxiv_id, {"arxiv_id": arxiv_id, "error": "not found"})
        return {"arxiv_id": arxiv_id, "error": "metadata file not found"}
