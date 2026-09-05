"""
Critic Agent — Phase 3.

Role: Fact-checker and supervisor. Validates the Synthesizer's answer against
the original evidence. Phase 3 improvements:
  - Structured JSON feedback (actionable, not just "answer is correct")
  - Missing aspects detection
  - Hallucination flag extraction
  - Proper source attribution passed through
"""

import re
import json
from src.agents.base import BaseAgent
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CriticAgent(BaseAgent):
    """
    Validates the synthesizer's answer against the evidence.

    Input : query (str), answer (str), extracted_passages (list[str]),
            doc_metadata (list[dict])
    Output: {
        "is_grounded": bool,
        "confidence": float,
        "feedback": str,
        "missing_aspects": list[str],
        "hallucination_flags": list[str],
        "sources": list[str]   ← properly attributed paper references
    }
    """

    def build_prompt(
        self,
        query: str,
        answer: str,
        extracted_passages: list,
        doc_metadata: list = None,
    ) -> str:
        evidence_block = ""
        for i, passage in enumerate(extracted_passages[:6]):
            meta = doc_metadata[i] if doc_metadata and i < len(doc_metadata) else {}
            arxiv_id = meta.get("arxiv_id", "unknown")
            evidence_block += f"[Evidence {i+1}] (arXiv:{arxiv_id}): {passage}\n\n"

        prompt = (
            f"<s>[INST] You are a strict academic fact-checker. "
            f"Verify whether the proposed answer is fully grounded in the evidence. "
            f"Check for: hallucinations (claims not in evidence), missing key points, "
            f"and unsupported statements.\n\n"
            f"Question: {query}\n\n"
            f"Evidence Passages:\n{evidence_block}"
            f"Proposed Answer:\n{answer}\n\n"
            f"Respond in EXACTLY this format (no extra text):\n"
            f"GROUNDED: <YES or NO>\n"
            f"CONFIDENCE: <number 0.0 to 1.0>\n"
            f"FEEDBACK: <one or two specific, actionable sentences>\n"
            f"MISSING: <comma-separated list of topics not covered, or NONE>\n"
            f"HALLUCINATIONS: <comma-separated list of unsupported claims, or NONE>\n"
            f"[/INST]"
        )
        return prompt

    def parse_output(self, raw_output: str, **kwargs) -> dict:
        doc_metadata = kwargs.get("doc_metadata", [])
        extracted_passages = kwargs.get("extracted_passages", [])

        is_grounded = True
        confidence = None
        feedback = "No specific feedback provided."
        missing_aspects = []
        hallucination_flags = []

        for line in raw_output.strip().split("\n"):
            clean_line = re.sub(r"[*_`]", "", line).strip()
            clean_line = re.sub(r"^[\s\-#>]+", "", clean_line).strip()

            m_grounded = re.search(r"\bGROUNDED\s*:\s*([A-Za-z]+)", clean_line, re.IGNORECASE)
            if m_grounded:
                verdict = m_grounded.group(1).upper()
                is_grounded = verdict.startswith("YES")
                continue

            m_conf = re.search(r"\bCONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)", clean_line, re.IGNORECASE)
            if m_conf:
                try:
                    c = float(m_conf.group(1))
                    confidence = max(0.0, min(1.0, c))
                except ValueError:
                    pass
                continue

            m_feedback = re.search(r"\bFEEDBACK\s*:\s*(.*)", clean_line, re.IGNORECASE)
            if m_feedback:
                fb = m_feedback.group(1).strip()
                if fb:
                    feedback = fb
                continue

            m_missing = re.search(r"\bMISSING\s*:\s*(.*)", clean_line, re.IGNORECASE)
            if m_missing:
                raw_missing = m_missing.group(1).strip()
                if raw_missing.upper() != "NONE" and raw_missing:
                    missing_aspects = [m.strip() for m in raw_missing.split(",") if m.strip() and m.strip().upper() != "NONE"]
                continue

            m_hall = re.search(r"\bHALLUCINATIONS\s*:\s*(.*)", clean_line, re.IGNORECASE)
            if m_hall:
                raw_hall = m_hall.group(1).strip()
                if raw_hall.upper() != "NONE" and raw_hall:
                    hallucination_flags = [h.strip() for h in raw_hall.split(",") if h.strip() and h.strip().upper() != "NONE"]
                continue

        # Whole-document fallback if line-by-line missed keys
        if confidence is None:
            m_conf = re.search(r"\bCONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)", raw_output, re.IGNORECASE)
            if m_conf:
                try:
                    confidence = max(0.0, min(1.0, float(m_conf.group(1))))
                except ValueError:
                    confidence = 0.85 if is_grounded else 0.4
            else:
                confidence = 0.85 if is_grounded else 0.4

        if "GROUNDED" in raw_output.upper():
            m_gr = re.search(r"\bGROUNDED\s*:\s*(YES|NO)\b", raw_output, re.IGNORECASE)
            if m_gr:
                is_grounded = m_gr.group(1).upper() == "YES"

        # ── Build proper source attribution ──────────────────────────────
        sources = []
        for i, meta in enumerate(doc_metadata):
            arxiv_id = meta.get("arxiv_id", "unknown")
            sources.append(f"arXiv:{arxiv_id}")

        logger.info(
            f"CriticAgent verdict — Grounded: {is_grounded}, "
            f"Confidence: {confidence:.2f}"
        )

        return {
            "is_grounded": is_grounded,
            "confidence": confidence,
            "feedback": feedback,
            "missing_aspects": missing_aspects,
            "hallucination_flags": hallucination_flags,
            "sources": sources,
        }

    def run(self, **kwargs) -> dict:
        query = kwargs.get("query", "")
        answer = kwargs.get("answer", "")
        extracted_passages = kwargs.get("extracted_passages", [])
        doc_metadata = kwargs.get("doc_metadata", [])

        prompt = self.build_prompt(
            query=query,
            answer=answer,
            extracted_passages=extracted_passages,
            doc_metadata=doc_metadata,
        )
        raw_output = self._generate(prompt)
        return self.parse_output(
            raw_output,
            doc_metadata=doc_metadata,
            extracted_passages=extracted_passages,
        )
