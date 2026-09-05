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

        # Multi-line / markdown block pattern:
        # Matches: "PASSAGE 1 (from Document X): ...", "**PASSAGE 1 (from Document X):** ...", etc.
        pattern = re.compile(
            r"(?:^|\n)\s*[*#-]*\s*(PASSAGE\s*\d+\s*(?:\([^\)]*\))?)\s*[*#:]*\s*[:\-]\s*([\s\S]+?)(?=(?:\n\s*[*#-]*\s*PASSAGE\s*\d+)|\Z)",
            re.IGNORECASE,
        )
        matches = list(pattern.finditer(raw))
        if matches:
            for m in matches:
                header = m.group(1).strip()
                body = m.group(2).strip().strip('"\'*')
                body = re.sub(r"^\s*[*#>-]+\s*", "", body).strip()
                if len(body) < 15:
                    continue
                doc_num_str = re.search(r"document\s*(\d+)", header, re.IGNORECASE)
                if not doc_num_str:
                    doc_num_str = re.search(r"document\s*(\d+)", body[:100], re.IGNORECASE)
                doc_num = int(doc_num_str.group(1)) - 1 if doc_num_str else (len(passages) % len(docs) if docs else 0)
                doc = docs[doc_num] if 0 <= doc_num < len(docs) else {}
                passages.append(body)
                sources.append(f"Document {doc_num + 1}")
                metadata.append({
                    "arxiv_id": doc.get("arxiv_id", "unknown"),
                    "doc_num": doc_num + 1,
                })
            if len(passages) >= 1:
                return passages, sources, metadata

        # Line-by-line fallback with markdown stripping
        for line in raw.strip().split("\n"):
            clean_line = re.sub(r"^[\s*\-#>]+", "", line).strip()
            if not re.match(r"^passage\s*\d+", clean_line, re.IGNORECASE):
                continue
            try:
                header, sep, text = clean_line.partition("):")
                if not sep:
                    header, sep, text = clean_line.partition(":")
                text = text.strip().strip('"\'*')
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
        blocks = [b.strip() for b in re.split(r"\n{2,}", raw.strip()) if b.strip()]
        for block in blocks:
            clean_block = re.sub(r"^\s*[*#-]+\s*", "", block).strip()
            m = re.match(r"^(?:passage|evidence|p)?\s*\[?(\d+)\]?[.):\s]+([\s\S]+)", clean_block, re.IGNORECASE)
            if m:
                text = m.group(2).strip().strip('"\'*')
                if len(text) < 20:
                    continue
                doc_ref = re.search(r"document\s*(\d+)", block, re.IGNORECASE)
                doc_num = int(doc_ref.group(1)) - 1 if doc_ref else (len(passages) % len(docs) if docs else 0)
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
        p1, s1, m1 = self._parse_structured(raw_output, retrieved_docs)
        if len(p1) >= MIN_PASSAGES:
            logger.info(f"ReaderAgent: strategy=structured, passages={len(p1)}")
            return {"extracted_passages": p1, "sources": s1, "doc_metadata": m1}

        # ── Try Strategy 2 ──
        p2, s2, m2 = self._parse_loose(raw_output, retrieved_docs)
        if len(p2) >= MIN_PASSAGES:
            logger.info(f"ReaderAgent: strategy=loose, passages={len(p2)}")
            return {"extracted_passages": p2, "sources": s2, "doc_metadata": m2}

        # If either structured or loose parsed at least 1 valid passage, preserve them
        best_p, best_s, best_m = (p1, s1, m1) if len(p1) >= len(p2) else (p2, s2, m2)
        if len(best_p) >= 1:
            logger.info(f"ReaderAgent: strategy=partial_structured, passages={len(best_p)}")
            return {"extracted_passages": best_p, "sources": best_s, "doc_metadata": best_m}

        # ── Strategy 3 fallback ──
        logger.warning(
            "ReaderAgent: strategies 1+2 yielded 0 passages. "
            "Falling back to TF-IDF sentence extraction."
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
