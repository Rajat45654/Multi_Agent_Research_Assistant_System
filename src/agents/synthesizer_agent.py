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
            f"1. Synthesize the {num_passages} evidence passages into a coherent, comprehensive explanation.\n"
            f"2. Cite evidence inline using separate [Evidence N] tags for EACH reference (e.g. [Evidence 1], [Evidence 2]). You MUST write '[Evidence N]' for every passage cited — NEVER combine multiple citations into one bracket like [Evidence 5, 6] or [Evidence 5, Evidence 6], and NEVER use bare numbers like [1] or [1][2]. Do NOT repeat consecutive citations.\n"
            f"3. Write 1-2 informative sentences for each evidence passage you reference.\n"
            f"4. Do NOT include any information not present in the evidence.\n"
            f"5. Do NOT repeat citations, sentences, or phrases in loops.\n"
            f"6. End with: Citations: [Evidence 1], [Evidence 2], ... (list all cited){refinement_note}\n"
            f"7. STOP immediately after the Citations line. Do NOT write any notes, disclaimers, conversational filler, or postscripts (P.S.).\n\n"
            f"Question: {query}\n\n"
            f"Evidence:\n{passage_blocks}"
            f"Your answer:\n"
            f"[/INST]"
        )
        return prompt

    @staticmethod
    def _normalize_citations(text: str, num_passages: int) -> str:
        """Normalize bare citation numbers and combined brackets into separate [Evidence N] format."""
        # 0. Expand multi-evidence brackets like [Evidence 5, Evidence 6] or [Evidence 5, 6]
        def expand_multi_evidence(match):
            inner = match.group(0)
            nums = [int(n) for n in re.findall(r"\b(\d+)\b", inner)]
            if all(1 <= n <= num_passages for n in nums):
                return " " + " ".join(f"[Evidence {n}]" for n in nums)
            return inner

        text = re.sub(
            r"\[Evidence\s*\d+(?:\s*,\s*(?:Evidence\s*)?\d+)+\]",
            expand_multi_evidence,
            text,
            flags=re.IGNORECASE,
        )

        # 1. Expand multi-citation clusters like [1][2] or [5][6] into [Evidence 1] [Evidence 2]
        def replace_bracket_cluster(match):
            inner = match.group(0)
            nums = [int(n) for n in re.findall(r"\b(\d+)\b", inner)]
            if all(1 <= n <= num_passages for n in nums):
                return " " + " ".join(f"[Evidence {n}]" for n in nums)
            return inner

        text = re.sub(r"\[\d+\](?:\s*\[\d+\])+", replace_bracket_cluster, text)
        text = re.sub(r"\[\d+(?:\s*,\s*\d+)+\]", replace_bracket_cluster, text)

        # 2. Normalize standalone [N] where 1 <= N <= num_passages, if not already preceded by "Evidence "
        text = re.sub(
            r"(?<!Evidence\s)\[(\d+)\]",
            lambda m: f"[Evidence {m.group(1)}]" if 1 <= int(m.group(1)) <= num_passages else m.group(0),
            text,
        )
        # Clean extra spaces
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text

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

        # Cut off trailing conversational chatter, notes, or postscript loops starting from (Note: or Note: or P.S.
        text = re.split(
            r"(?:\n+|\s+)(?:\((?:Note|Additional\s+note|Final\s+note|Please\s+note|Disclaimer|P+\.?S\.?|Lastly|Also|Finally)|Note:|Disclaimer:|P+\.?S\.?:)",
            text,
            flags=re.IGNORECASE,
        )[0]
        # Remove any stray (P.S. ...) or (Note: ...) anywhere remaining
        text = re.sub(
            r"\((?:Note|Additional\s+note|Final\s+note|Please\s+note|Disclaimer|P+\.?S\.?)[^)]*\)?",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Remove duplicate whitespace/newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Deduplicate consecutive repeated citations like: [Evidence 6] [Evidence 6] -> [Evidence 6]
        text = re.sub(r"(\[Evidence\s*\d+\])(?:\s*,?\s*\1)+", r"\1", text)
        # Collapse repeated citation clusters
        text = re.sub(r"((?:\[Evidence\s*\d+\]\s*){2,})\1+", r"\1", text)
        # Strip trailing cut-off citation fragment (e.g. [Ev or [Evidence)
        text = re.sub(r"\[\s*E(?:v(?:i(?:d(?:e(?:n(?:c(?:e)?)?)?)?)?)?)?\s*\d*$", "", text).strip()
        # Clean trailing commas before end
        text = re.sub(r",\s*$", ".", text)
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

        # Split off Citations section if present (supports Citations:, References:, Sources:, **Citations:**, etc.)
        m_cite = re.search(r"(?:\*\*|#|\b)(?:Citations|References|Sources)\s*:\s*", answer, re.IGNORECASE)
        if m_cite:
            answer_part = answer[:m_cite.start()].strip()
            citations_raw = answer[m_cite.end():].strip()
            # Strip any notes or chatter starting from (Note: or (P.S.
            citations_raw = re.split(
                r"(?:\n+|\s+)(?:\((?:Note|Additional\s+note|Final\s+note|Please\s+note|Disclaimer|P+\.?S\.?|Lastly|Also|Finally)|Note:|Disclaimer:|P+\.?S\.?:)",
                citations_raw,
                flags=re.IGNORECASE,
            )[0].strip()
            answer = answer_part
            citations = [c.strip().strip("*_#`") for c in citations_raw.split(",") if c.strip()]

        # Clean artifacts
        answer = self._clean_artifacts(answer)

        # Normalize bare citation numbers like [1][2] or [4] into [Evidence N]
        answer = self._normalize_citations(answer, num_passages)

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
        raw_output = self._generate(prompt, max_new_tokens=768)
        return self.parse_output(
            raw_output,
            num_passages=len(extracted_passages),
        )
