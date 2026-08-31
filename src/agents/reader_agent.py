"""
Reader Agent.

Role: Given a user query and a list of retrieved document chunks,
extract ONLY the passages that are directly relevant to answering the question.
Filters out noise so the Synthesizer only sees clean, targeted evidence.
"""

from src.agents.base import BaseAgent
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReaderAgent(BaseAgent):
    """
    Extracts relevant passages from retrieved document chunks.
    
    Input : query (str), retrieved_docs (list of chunk dicts)
    Output: {"extracted_passages": [...], "sources": [...]}
    """

    def build_prompt(self, query: str, retrieved_docs: list) -> str:
        # Format each retrieved document chunk into numbered blocks
        doc_blocks = ""
        for i, doc in enumerate(retrieved_docs[:8]):  # Cap at 8 docs
            doc_blocks += (
                f"[Document {i+1}] (Paper ID: {doc.get('arxiv_id', 'unknown')})\n"
                f"{doc.get('text', '')[:800]}\n\n"
            )

        prompt = (
            f"<s>[INST] You are a precise research reading assistant. "
            f"Your task is to extract the EXACT passages from the documents below "
            f"that are most relevant to answering the given question. "
            f"Do NOT paraphrase or add any information not present in the documents. "
            f"For each extracted passage, note which Document number it came from.\n\n"
            f"Question: {query}\n\n"
            f"Documents:\n{doc_blocks}"
            f"Output format:\n"
            f"PASSAGE 1 (from Document X): <exact passage text>\n"
            f"PASSAGE 2 (from Document Y): <exact passage text>\n"
            f"(extract up to 5 most relevant passages)\n"
            f"[/INST]"
        )
        return prompt

    def parse_output(self, raw_output: str) -> dict:
        passages = []
        sources = []

        for line in raw_output.strip().split("\n"):
            line = line.strip()
            if line.lower().startswith("passage"):
                # Try to extract "PASSAGE N (from Document M): text"
                try:
                    header, _, text = line.partition("):")
                    doc_ref = header.split("Document")[-1].strip().rstrip(")")
                    passages.append(text.strip())
                    sources.append(f"Document {doc_ref}")
                except Exception:
                    passages.append(line)
                    sources.append("unknown")

        if not passages:
            # Fallback: just return the raw output if parsing fails
            logger.warning("ReaderAgent: Could not parse structured passages. Using raw output.")
            passages = [raw_output.strip()]
            sources = ["unknown"]

        logger.info(f"ReaderAgent extracted {len(passages)} passages.")
        return {"extracted_passages": passages, "sources": sources}
