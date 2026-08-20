"""
ArXiv API collector.

Fetches papers based on specified categories and downloads PDFs.
Saves metadata to data/raw/metadata.jsonl.
"""

import json
import time
import arxiv
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.config import Config, load_config

logger = get_logger(__name__)

class ArxivCollector:
    """Collects papers from ArXiv and downloads PDFs."""
    
    def __init__(self, config: Config):
        self.config = config
        self.raw_dir = Path(config.paths.raw_dir)
        self.papers_dir = Path(config.paths.papers_dir)
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.raw_dir / "metadata.jsonl"

    def fetch_papers(self) -> None:
        """Fetch papers from ArXiv based on config."""
        logger.info(f"Starting paper collection. Target: {self.config.data_collection.num_papers} papers.")
        
        categories = self.config.data_collection.categories
        query = " OR ".join([f"cat:{c}" for c in categories])
        
        client = arxiv.Client(
            page_size=self.config.data_collection.max_results_per_query,
            delay_seconds=3.0,
            num_retries=3
        )
        
        search = arxiv.Search(
            query=query,
            max_results=self.config.data_collection.num_papers,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        metadata_records: List[Dict[str, Any]] = []
        downloaded = 0
        
        for result in tqdm(client.results(search), total=self.config.data_collection.num_papers, desc="Fetching ArXiv"):
            arxiv_id = result.get_short_id()
            pdf_path = self.papers_dir / f"{arxiv_id}.pdf"
            
            # Skip if already downloaded
            if pdf_path.exists():
                logger.debug(f"Paper {arxiv_id} already exists, skipping download.")
                continue
            
            try:
                import urllib.request
                urllib.request.urlretrieve(result.pdf_url, str(pdf_path))
                downloaded += 1
                
                metadata = {
                    "arxiv_id": arxiv_id,
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "abstract": result.summary,
                    "published": result.published.isoformat(),
                    "url": result.entry_id,
                    "categories": result.categories
                }
                metadata_records.append(metadata)
                
                # Append to metadata file incrementally
                with open(self.metadata_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(metadata) + "\n")
                    
            except Exception as e:
                logger.error(f"Failed to download {arxiv_id}: {e}")
                
            time.sleep(1.0) # Be nice to ArXiv API
            
        logger.info(f"Collection complete. Downloaded {downloaded} new papers.")

if __name__ == "__main__":
    cfg = load_config()
    collector = ArxivCollector(cfg)
    collector.fetch_papers()
