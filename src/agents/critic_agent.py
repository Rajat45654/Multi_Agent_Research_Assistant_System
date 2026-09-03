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
        confidence = 0.7
        feedback = "No specific feedback provided."
        missing_aspects = []
        hallucination_flags = []

        for line in raw_output.strip().split("\n"):
            line = line.strip()
            upper = line.upper()

            if upper.startswith("GROUNDED:"):
                verdict = line.split(":", 1)[-1].strip().upper()
                is_grounded = verdict.startswith("YES")

            elif upper.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[-1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass

            elif upper.startswith("FEEDBACK:"):
                feedback = line.split(":", 1)[-1].strip()

            elif upper.startswith("MISSING:"):
                raw_missing = line.split(":", 1)[-1].strip()
                if raw_missing.upper() != "NONE" and raw_missing:
                    missing_aspects = [m.strip() for m in raw_missing.split(",") if m.strip()]

            elif upper.startswith("HALLUCINATIONS:"):
                raw_hall = line.split(":", 1)[-1].strip()
                if raw_hall.upper() != "NONE" and raw_hall:
                    hallucination_flags = [h.strip() for h in raw_hall.split(",") if h.strip()]

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
