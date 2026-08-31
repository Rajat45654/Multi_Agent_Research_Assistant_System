"""
Synthesizer Agent.

Role: Takes the extracted passages from the ReaderAgent and combines them
into a single, coherent, well-cited answer to the original question.
"""

from src.agents.base import BaseAgent
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SynthesizerAgent(BaseAgent):
    """
    Synthesizes a final answer from extracted passages.

    Input : query (str), extracted_passages (list[str]), sources (list[str])
    Output: {"answer": str, "citations": list[str]}
    """

    def build_prompt(self, query: str, extracted_passages: list, sources: list) -> str:
        passage_blocks = ""
        for i, (passage, source) in enumerate(zip(extracted_passages, sources)):
            passage_blocks += f"[Evidence {i+1}] ({source}): {passage}\n\n"

        prompt = (
            f"<s>[INST] You are an expert academic research assistant. "
            f"Using ONLY the evidence passages provided below, write a clear, "
            f"comprehensive answer to the question. "
            f"You MUST cite your sources using the format [Evidence N]. "
            f"Do NOT include any information not present in the evidence.\n\n"
            f"Question: {query}\n\n"
            f"Evidence:\n{passage_blocks}"
            f"Provide your answer below. End with a 'Citations:' section listing all evidence used.\n"
            f"[/INST]"
        )
        return prompt

    def parse_output(self, raw_output: str) -> dict:
        answer = raw_output.strip()
        citations = []

        # Split off citations section if present
        if "Citations:" in answer:
            parts = answer.split("Citations:", 1)
            answer = parts[0].strip()
            citations_raw = parts[1].strip()
            citations = [c.strip() for c in citations_raw.split("\n") if c.strip()]

        logger.info(f"SynthesizerAgent produced answer ({len(answer)} chars), {len(citations)} citations.")
        return {"answer": answer, "citations": citations}
