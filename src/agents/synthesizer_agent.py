"""
Synthesizer Agent — Phase 3.

Role: Takes extracted passages from the ReaderAgent and synthesizes a
comprehensive, well-cited answer. Phase 3 improvements:
  - Explicit prompt requiring ALL passages to be used
  - Post-processing cleanup of LLM formatting artifacts ([/ ], [/Evidence N])
  - Minimum detail level enforced (2+ sentences per passage)
  - Citation cross-check: verifies [Evidence N] tags are valid
  - Accepts optional critic_feedback for iterative refinement (iteration 2+)
"""

import re
from src.agents.base import BaseAgent
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SynthesizerAgent(BaseAgent):
    """
    Synthesizes a final answer from extracted passages.

    Input : query (str), extracted_passages (list[str]), sources (list[str]),
            doc_metadata (list[dict]), critic_feedback (str, optional)
    Output: {"answer": str, "citations": list[str], "citation_ids": list[int]}
    """

    def build_prompt(
        self,
        query: str,
        extracted_passages: list,
        sources: list,
        doc_metadata: list = None,
        critic_feedback: str = "",
    ) -> str:
        passage_blocks = ""
        for i, (passage, source) in enumerate(zip(extracted_passages, sources)):
            meta = doc_metadata[i] if doc_metadata and i < len(doc_metadata) else {}
            arxiv_id = meta.get("arxiv_id", "unknown")
            passage_blocks += f"[Evidence {i+1}] (Source: {arxiv_id}): {passage}\n\n"

        num_passages = len(extracted_passages)
        refinement_note = ""
        if critic_feedback:
            refinement_note = (
                f"\n\nIMPORTANT — Your previous answer was rejected for this reason:\n"
                f"{critic_feedback}\n"
                f"Address these issues specifically in your new answer.\n"
            )

        prompt = (
            f"<s>[INST] You are an expert academic research assistant. "
            f"Using ONLY the {num_passages} evidence passages provided below, "
            f"write a clear, comprehensive, detailed answer to the question. "
            f"Rules:\n"
            f"1. You MUST use ALL {num_passages} evidence passages in your answer.\n"
            f"2. Cite each piece of evidence using [Evidence N] format inline.\n"
            f"3. Write at least 2 sentences for each evidence passage you use.\n"
            f"4. Do NOT include any information not present in the evidence.\n"
            f"5. Do NOT use formatting like [/ ], [/Evidence], or markdown headers.\n"
            f"6. End with: Citations: [Evidence 1], [Evidence 2], ... (list all used){refinement_note}\n\n"
            f"Question: {query}\n\n"
            f"Evidence:\n{passage_blocks}"
            f"Your answer:\n"
            f"[/INST]"
        )
        return prompt

    @staticmethod
    def _clean_artifacts(text: str) -> str:
        """Remove common LLM output artifacts from the synthesizer's response."""
        # Strip instruction markers like [/INST], [INST]
        text = re.sub(r"\[/?INST\]", "", text, flags=re.IGNORECASE)
        # Strip special sentence markers <s>, </s>
        text = re.sub(r"<\/?s>", "", text)
        # Remove [/ ], [/Evidence N], [/ Evidence N] patterns
        text = re.sub(r"\[/\s*(?:Evidence\s*\d+)?\s*\]", "", text)
        # Remove standalone [/ ] noise
        text = re.sub(r"\[/\s*\]", "", text)
        # Normalize [Evidence N] — ensure no spaces inside brackets
        text = re.sub(r"\[\s*Evidence\s*(\d+)\s*\]", r"[Evidence \1]", text)
        # Strip leading "Answer:" or "Response:" prefix if regurgitated
        text = re.sub(r"^(?:Answer|Response|Explanation):\s*", "", text.strip(), flags=re.IGNORECASE)
        # Remove duplicate whitespace/newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Strip dangling trailing asterisks or markdown artifacts
        text = re.sub(r"[\s*#\-]+$", "", text)
        return text.strip()

    def _verify_citations(self, answer: str, num_passages: int) -> list[int]:
        """Extract which Evidence IDs are actually referenced in the answer."""
        found = re.findall(r"\[Evidence\s*(\d+)\]", answer)
        valid = [int(n) for n in found if 1 <= int(n) <= num_passages]
        return sorted(set(valid))

    def parse_output(self, raw_output: str, **kwargs) -> dict:
        num_passages = kwargs.get("num_passages", 10)
        answer = raw_output.strip()
        citations = []

        # Split off Citations section if present (supports markdown: **Citations:**, Citations:, etc.)
        m_cite = re.search(r"(?:\*\*|#|\b)Citations\s*:\s*", answer, re.IGNORECASE)
        if m_cite:
            answer_part = answer[:m_cite.start()].strip()
            citations_raw = answer[m_cite.end():].strip()
            answer = answer_part
            citations = [c.strip().strip("*_#`") for c in citations_raw.split(",") if c.strip()]

        # Clean artifacts
        answer = self._clean_artifacts(answer)

        # Verify which evidence IDs are actually used
        citation_ids = self._verify_citations(answer, num_passages)

        logger.info(
            f"SynthesizerAgent produced answer ({len(answer)} chars), "
            f"citations={citation_ids}"
        )
        return {
            "answer": answer,
            "citations": citations,
            "citation_ids": citation_ids,
        }

    def run(self, **kwargs) -> dict:
        query = kwargs.get("query", "")
        extracted_passages = kwargs.get("extracted_passages", [])
        sources = kwargs.get("sources", [])
        doc_metadata = kwargs.get("doc_metadata", None)
        critic_feedback = kwargs.get("critic_feedback", "")

        prompt = self.build_prompt(
            query=query,
            extracted_passages=extracted_passages,
            sources=sources,
            doc_metadata=doc_metadata,
            critic_feedback=critic_feedback,
        )
        raw_output = self._generate(prompt)
        return self.parse_output(
            raw_output,
            num_passages=len(extracted_passages),
        )
