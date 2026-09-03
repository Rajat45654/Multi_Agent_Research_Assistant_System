"""
Reader Agent — Phase 3 (Robust Multi-Strategy Parser).

Role: Given a user query and retrieved document chunks, extract the most
relevant passages. Uses three parsing strategies in priority order so that
format variations from the v2 fine-tuned model never result in a single-
passage fallback.

Parsing strategies (tried in order):
  1. Structured   — "PASSAGE N (from Document X): text"
  2. Looser regex — numbered variants: "1.", "[1]", "Passage 1:", "Evidence 1:"
  3. Sentence TF-IDF — split raw output into sentences, score by query overlap,
                        return top-N most relevant
"""

import re
from src.agents.base import BaseAgent
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Min/max passage counts
MIN_PASSAGES = 3
MAX_PASSAGES = 6


class ReaderAgent(BaseAgent):
    """
    Extracts relevant passages from retrieved document chunks.

    Input : query (str), retrieved_docs (list of chunk dicts)
    Output: {"extracted_passages": [...], "sources": [...], "doc_metadata": [...]}
    """

    def build_prompt(self, query: str, retrieved_docs: list) -> str:
        doc_blocks = ""
        for i, doc in enumerate(retrieved_docs[:8]):
            doc_blocks += (
                f"[Document {i+1}] (Paper ID: {doc.get('arxiv_id', 'unknown')})\n"
                f"{doc.get('text', '')[:800]}\n\n"
            )

        prompt = (
            f"<s>[INST] You are a precise research reading assistant. "
            f"Extract the {MIN_PASSAGES} to {MAX_PASSAGES} MOST relevant passages "
            f"from the documents below that directly answer the question. "
            f"Copy passages EXACTLY from the documents — do not paraphrase. "
            f"Note which Document number each passage came from.\n\n"
            f"Question: {query}\n\n"
            f"Documents:\n{doc_blocks}"
            f"Output format (use EXACTLY this format, one passage per line):\n"
            f"PASSAGE 1 (from Document X): <exact passage text>\n"
            f"PASSAGE 2 (from Document Y): <exact passage text>\n"
            f"PASSAGE 3 (from Document Z): <exact passage text>\n"
            f"[/INST]"
        )
        return prompt

    # ── Strategy 1: strict structured parser ─────────────────────────────
    def _parse_structured(self, raw: str, docs: list) -> tuple[list, list, list]:
        passages, sources, metadata = [], [], []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line.lower().startswith("passage"):
                continue
            try:
                header, _, text = line.partition("):")
                text = text.strip()
                if not text:
                    continue
                doc_num_str = re.search(r"document\s*(\d+)", header, re.IGNORECASE)
                doc_num = int(doc_num_str.group(1)) - 1 if doc_num_str else 0
                doc = docs[doc_num] if 0 <= doc_num < len(docs) else {}
                passages.append(text)
                sources.append(f"Document {doc_num + 1}")
                metadata.append({
                    "arxiv_id": doc.get("arxiv_id", "unknown"),
                    "doc_num": doc_num + 1,
                })
            except Exception:
                continue
        return passages, sources, metadata

    # ── Strategy 2: loose numbered-list parser ────────────────────────────
    def _parse_loose(self, raw: str, docs: list) -> tuple[list, list, list]:
        passages, sources, metadata = [], [], []
        # Matches: "1.", "[1]", "Passage 1:", "Evidence 1:", "1)"
        pattern = re.compile(
            r"(?:passage|evidence|p)?\s*\[?(\d+)\]?[.):\s]+(.+)",
            re.IGNORECASE,
        )
        # Also try splitting by blank lines as paragraph chunks
        blocks = [b.strip() for b in re.split(r"\n{2,}", raw.strip()) if b.strip()]
        for block in blocks:
            m = pattern.match(block)
            if m:
                text = m.group(2).strip()
                if len(text) < 20:          # skip trivially short matches
                    continue
                # Try to extract doc reference from block
                doc_ref = re.search(r"document\s*(\d+)", block, re.IGNORECASE)
                doc_num = int(doc_ref.group(1)) - 1 if doc_ref else 0
                doc = docs[doc_num] if 0 <= doc_num < len(docs) else {}
                passages.append(text)
                sources.append(f"Document {doc_num + 1}")
                metadata.append({
                    "arxiv_id": doc.get("arxiv_id", "unknown"),
                    "doc_num": doc_num + 1,
                })
        return passages, sources, metadata

    # ── Strategy 3: sentence TF-IDF fallback ─────────────────────────────
    def _parse_sentence_tfidf(
        self, raw: str, query: str, docs: list
    ) -> tuple[list, list, list]:
        """
        Score every sentence in the raw output by term overlap with the query,
        return the top MIN_PASSAGES highest-scoring ones.
        """
        query_terms = set(re.findall(r"\w+", query.lower()))
        sentences = re.split(r"(?<=[.!?])\s+", raw.strip())
        scored = []
        for sent in sentences:
            if len(sent) < 30:
                continue
            terms = set(re.findall(r"\w+", sent.lower()))
            overlap = len(query_terms & terms) / max(len(query_terms), 1)
            scored.append((overlap, sent))
        scored.sort(key=lambda x: -x[0])
        top_sents = [s for _, s in scored[:MAX_PASSAGES]]
        # Best-effort: attribute to doc 1 as we lost track during fallback
        passages = top_sents
        sources = ["Document 1"] * len(top_sents)
        doc = docs[0] if docs else {}
        metadata = [{"arxiv_id": doc.get("arxiv_id", "unknown"), "doc_num": 1}] * len(top_sents)
        return passages, sources, metadata

    def parse_output(self, raw_output: str, **kwargs) -> dict:
        retrieved_docs = kwargs.get("retrieved_docs", [])
        query = kwargs.get("query", "")

        # ── Try Strategy 1 ──
        passages, sources, metadata = self._parse_structured(raw_output, retrieved_docs)
        if len(passages) >= MIN_PASSAGES:
            logger.info(f"ReaderAgent: strategy=structured, passages={len(passages)}")
            return {"extracted_passages": passages, "sources": sources, "doc_metadata": metadata}

        # ── Try Strategy 2 ──
        passages, sources, metadata = self._parse_loose(raw_output, retrieved_docs)
        if len(passages) >= MIN_PASSAGES:
            logger.info(f"ReaderAgent: strategy=loose, passages={len(passages)}")
            return {"extracted_passages": passages, "sources": sources, "doc_metadata": metadata}

        # ── Strategy 3 fallback ──
        logger.warning(
            f"ReaderAgent: strategies 1+2 yielded only {len(passages)} passages. "
            f"Falling back to TF-IDF sentence extraction."
        )
        passages, sources, metadata = self._parse_sentence_tfidf(raw_output, query, retrieved_docs)
        logger.info(f"ReaderAgent: strategy=tfidf_fallback, passages={len(passages)}")
        return {"extracted_passages": passages, "sources": sources, "doc_metadata": metadata}

    def run(self, **kwargs) -> dict:
        """Override run to pass retrieved_docs and query into parse_output."""
        query = kwargs.get("query", "")
        retrieved_docs = kwargs.get("retrieved_docs", [])
        prompt = self.build_prompt(query, retrieved_docs)
        raw_output = self._generate(prompt)
        return self.parse_output(raw_output, retrieved_docs=retrieved_docs, query=query)
