"""
Query Decomposer & Multi-Hop Context Fusion — Phase 5.

Role: Analyzes user research questions to identify multi-faceted, comparative,
or compound topics (e.g. "Compare LoRA and QLoRA in terms of memory efficiency",
"What are the differences between FlashAttention and standard self-attention?").

For multi-faceted queries, it decomposes the question into targeted sub-queries,
executes sub-retrievals, and fuses the results with Reciprocal Rank Fusion (RRF)
so that all facets have balanced document representation.
"""

import re
from typing import List, Dict, Tuple
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

COMPARATIVE_TRIGGERS = [
    r"\bcompare\b",
    r"\bcontrast\b",
    r"\bdifference(?:s)?\s+between\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\btrade-?offs?\s+between\b",
    r"\bhow\s+does\s+.+\s+differ\s+from\b",
    r"\badvantage(?:s)?\s+of\s+.+\s+over\b",
    r"\bboth\s+.+\s+and\b",
]


class QueryDecomposer:
    """
    Decomposes multi-faceted and comparative research queries into sub-queries,
    and fuses multi-hop retrieval results using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def is_multi_hop(self, query: str) -> bool:
        """Determines if a query warrants multi-hop / comparative decomposition."""
        query_lower = query.lower()
        for pattern in COMPARATIVE_TRIGGERS:
            if re.search(pattern, query_lower):
                return True
        # Check for compound comparisons like "A vs B" or "X and Y"
        if re.search(r"\b[\w-]+\s+(?:vs\.?|versus)\s+[\w-]+\b", query_lower):
            return True
        return False

    def decompose(self, query: str) -> List[str]:
        """
        Decomposes query into 2-3 focused sub-queries.
        Uses fast heuristic splitting with fallback to original query.
        """
        if not self.is_multi_hop(query):
            return [query]

        sub_queries = []
        q_lower = query.lower().strip("?. ")

        # Pattern 1: "Compare X and Y (in terms of Z)" or "Compare X vs Y"
        m_comp = re.search(
            r"compare\s+(?:the\s+)?(.+?)\s+(?:and|with|versus|vs\.?)\s+(.+?)(?:\s+(?:in\s+terms\s+of|regarding|with\s+respect\s+to|for)\s+(.+))?$",
            q_lower,
            re.IGNORECASE,
        )
        if m_comp:
            item_a = m_comp.group(1).strip()
            item_b = m_comp.group(2).strip()
            aspect = m_comp.group(3).strip() if m_comp.group(3) else ""

            aspect_suffix = f" {aspect}" if aspect else ""
            sub_queries.append(f"{item_a}{aspect_suffix}")
            sub_queries.append(f"{item_b}{aspect_suffix}")
            return [q.strip() for q in sub_queries if len(q.strip()) > 3]

        # Pattern 2: "Differences between X and Y"
        m_diff = re.search(
            r"difference(?:s)?\s+between\s+(.+?)\s+and\s+(.+?)(?:\s+(?:in|for|regarding)\s+(.+))?$",
            q_lower,
            re.IGNORECASE,
        )
        if m_diff:
            item_a = m_diff.group(1).strip()
            item_b = m_diff.group(2).strip()
            aspect = m_diff.group(3).strip() if m_diff.group(3) else ""

            aspect_suffix = f" {aspect}" if aspect else ""
            sub_queries.append(f"{item_a}{aspect_suffix}")
            sub_queries.append(f"{item_b}{aspect_suffix}")
            return [q.strip() for q in sub_queries if len(q.strip()) > 3]

        # Pattern 3: "X vs Y" or "X versus Y"
        m_vs = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)$", q_lower, re.IGNORECASE)
        if m_vs:
            item_a = m_vs.group(1).strip()
            item_b = m_vs.group(2).strip()
            sub_queries.append(item_a)
            sub_queries.append(item_b)
            return [q.strip() for q in sub_queries if len(q.strip()) > 3]

        # Default fallback: return original query
        return [query]

    @staticmethod
    def fuse_results(
        retrieved_lists: List[List[Dict]], top_k: int = 10, rrf_k: int = 60
    ) -> List[Dict]:
        """
        Fuses multiple ranked document lists into a single ranked list using
        Reciprocal Rank Fusion (RRF):
            Score(d) = sum(1 / (rrf_k + rank_i(d)))
        Ensures fair, balanced representation across all query facets.
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}

        for doc_list in retrieved_lists:
            for rank, doc in enumerate(doc_list):
                # Unique key: chunk id or fallback to text hash / snippet
                doc_key = doc.get("chunk_id") or f"{doc.get('arxiv_id', 'unknown')}_{hash(doc.get('text', '')[:100])}"
                doc_map[doc_key] = doc
                scores[doc_key] = scores.get(doc_key, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort by fused score descending
        ranked_keys = sorted(scores.keys(), key=lambda k: -scores[k])
        fused = [doc_map[k] for k in ranked_keys[:top_k]]
        return fused
