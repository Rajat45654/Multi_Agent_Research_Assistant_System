"""
Critic Agent.

Role: Acts as a fact-checker / supervisor. It reads the Synthesizer's proposed
answer alongside the original evidence passages and determines:
  1. Whether the answer is grounded in the evidence (no hallucinations).
  2. A confidence score from 0.0 to 1.0.
  3. Specific feedback if the answer needs to be improved.
"""

from src.agents.base import BaseAgent
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CriticAgent(BaseAgent):
    """
    Validates the synthesizer's answer against the evidence.

    Input : query (str), answer (str), extracted_passages (list[str])
    Output: {"is_grounded": bool, "confidence": float, "feedback": str}
    """

    def build_prompt(self, query: str, answer: str, extracted_passages: list) -> str:
        evidence_block = "\n\n".join(
            [f"[Evidence {i+1}]: {p}" for i, p in enumerate(extracted_passages[:5])]
        )

        prompt = (
            f"<s>[INST] You are a strict academic fact-checker. "
            f"Your task is to verify whether the proposed answer to a question "
            f"is fully supported by the provided evidence passages. "
            f"Check for hallucinations (claims not supported by evidence), "
            f"missing key points, and inaccuracies.\n\n"
            f"Question: {query}\n\n"
            f"Evidence Passages:\n{evidence_block}\n\n"
            f"Proposed Answer:\n{answer}\n\n"
            f"Respond in EXACTLY this format:\n"
            f"GROUNDED: <YES or NO>\n"
            f"CONFIDENCE: <a number between 0.0 and 1.0>\n"
            f"FEEDBACK: <one or two sentences of specific feedback>\n"
            f"[/INST]"
        )
        return prompt

    def parse_output(self, raw_output: str) -> dict:
        is_grounded = True
        confidence = 0.7
        feedback = "No feedback provided."

        for line in raw_output.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("GROUNDED:"):
                verdict = line.split(":", 1)[-1].strip().upper()
                is_grounded = verdict == "YES"
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[-1].strip())
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                except ValueError:
                    pass
            elif line.upper().startswith("FEEDBACK:"):
                feedback = line.split(":", 1)[-1].strip()

        logger.info(
            f"CriticAgent verdict — Grounded: {is_grounded}, "
            f"Confidence: {confidence:.2f}"
        )
        return {
            "is_grounded": is_grounded,
            "confidence": confidence,
            "feedback": feedback,
        }
