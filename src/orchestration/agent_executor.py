"""
Agent Executor — Phase 3 (Iterative Refinement + Source Attribution).

Orchestrates the full multi-agent pipeline:

    User Query
        → [RetrievalTool]      retrieve top-K chunks
        → [ReaderAgent]        extract 3-6 relevant passages (3-strategy parser)
        → [SynthesizerAgent]   write cited answer using ALL passages
        → [CriticAgent]        validate, score, produce structured feedback
        → If confidence < threshold:
              Synthesizer gets specific feedback → tries again (max 3 iterations)
        → Return best answer + full reasoning trace + proper source attribution

Design: Pure Python state machine — no LangChain overhead.
Full traceability: every step logged to reasoning_trace.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.agents.reader_agent import ReaderAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.agents.critic_agent import CriticAgent
from src.tools.retrieval_tool import RetrievalTool
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """The final structured response returned to the user / API."""
    answer: str
    sources: List[str]          # Properly attributed: ["arXiv:XXXX.XXXXX", ...]
    citations: List[str]        # Citation strings from synthesizer
    citation_ids: List[int]     # Verified [Evidence N] IDs used in answer
    confidence: float
    is_grounded: bool
    missing_aspects: List[str]
    hallucination_flags: List[str]
    reasoning_trace: List[Dict[str, Any]]
    iterations: int
    query: str


@dataclass
class ConversationMemory:
    """Maintains a sliding window of conversation history."""
    max_window: int = 10
    history: List[Dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_window:
            self.history = self.history[-self.max_window:]

    def get_context(self) -> str:
        return "\n".join([f"{h['role'].upper()}: {h['content']}" for h in self.history])


class AgentExecutor:
    """
    Orchestrates the multi-agent research pipeline (Phase 3).

    Usage:
        executor = AgentExecutor(cfg)
        response = executor.query("What is the transformer attention mechanism?")
        print(response.answer)
        print(response.sources)        # Now shows arXiv IDs, not "unknown"
        print(response.reasoning_trace)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.max_iterations = cfg.agents.max_iterations
        self.confidence_threshold = cfg.agents.confidence_threshold

        logger.info("Initializing AgentExecutor...")

        self.retrieval_tool = RetrievalTool(cfg)
        self.reader = ReaderAgent(cfg)
        self.synthesizer = SynthesizerAgent(cfg)
        self.critic = CriticAgent(cfg)
        self.memory = ConversationMemory(max_window=cfg.agents.memory_window)

        logger.info("AgentExecutor ready.")

    def query(self, question: str, top_k: int = 10) -> AgentResponse:
        """
        Runs the full multi-agent pipeline for a given question.

        Args:
            question: The user's natural language research question.
            top_k   : Number of documents to retrieve for context.

        Returns:
            AgentResponse with answer, sources, confidence, and full trace.
        """
        reasoning_trace = []
        self.memory.add("user", question)

        # ── Step 1: Retrieval ─────────────────────────────────────────────
        logger.info(f"[Step 1/4] Retrieval: '{question[:80]}'")
        retrieved_docs = self.retrieval_tool.retrieve(question, top_k=top_k)
        reasoning_trace.append({
            "step": "retrieval",
            "num_docs_retrieved": len(retrieved_docs),
            "top_doc_ids": [d.get("arxiv_id", "?") for d in retrieved_docs[:3]],
        })

        # ── Step 2: Reader ────────────────────────────────────────────────
        logger.info("[Step 2/4] ReaderAgent: extracting relevant passages...")
        reader_output = self.reader.run(
            query=question,
            retrieved_docs=retrieved_docs,
        )
        extracted_passages = reader_output["extracted_passages"]
        sources = reader_output["sources"]
        doc_metadata = reader_output.get("doc_metadata", [])

        reasoning_trace.append({
            "step": "reader",
            "num_passages_extracted": len(extracted_passages),
            "parse_strategy": reader_output.get("strategy", "unknown"),
            "passages_preview": [p[:100] for p in extracted_passages[:3]],
        })

        # ── Steps 3+4: Synthesizer + Critic iterative loop ────────────────
        answer = ""
        citations = []
        citation_ids = []
        confidence = 0.0
        is_grounded = False
        missing_aspects = []
        hallucination_flags = []
        critic_feedback = ""

        # Track best answer in case we exhaust iterations without approval
        best_answer = ""
        best_confidence = 0.0
        best_citations = []
        best_citation_ids = []
        best_missing = []
        best_flags = []

        iteration = 0
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"[Step 3/4] SynthesizerAgent: iteration {iteration}...")

            synth_output = self.synthesizer.run(
                query=question,
                extracted_passages=extracted_passages,
                sources=sources,
                doc_metadata=doc_metadata,
                critic_feedback=critic_feedback,   # empty on iter 1, populated on retries
            )
            answer = synth_output["answer"]
            citations = synth_output["citations"]
            citation_ids = synth_output["citation_ids"]

            reasoning_trace.append({
                "step": f"synthesizer_iteration_{iteration}",
                "answer_preview": answer[:200],
                "num_citations": len(citations),
                "citation_ids": citation_ids,
            })

            # ── Critic validation ─────────────────────────────────────────
            logger.info(f"[Step 4/4] CriticAgent: validating iteration {iteration}...")
            critic_output = self.critic.run(
                query=question,
                answer=answer,
                extracted_passages=extracted_passages,
                doc_metadata=doc_metadata,
            )
            is_grounded = critic_output["is_grounded"]
            confidence = critic_output["confidence"]
            critic_feedback = critic_output["feedback"]
            missing_aspects = critic_output.get("missing_aspects", [])
            hallucination_flags = critic_output.get("hallucination_flags", [])
            attributed_sources = critic_output.get("sources", sources)

            reasoning_trace.append({
                "step": f"critic_iteration_{iteration}",
                "is_grounded": is_grounded,
                "confidence": confidence,
                "feedback": critic_feedback,
                "missing_aspects": missing_aspects,
                "hallucination_flags": hallucination_flags,
            })

            # Track best result so far
            if confidence > best_confidence:
                best_confidence = confidence
                best_answer = answer
                best_citations = citations
                best_citation_ids = citation_ids
                best_missing = missing_aspects
                best_flags = hallucination_flags

            # Approve if grounded and confident enough
            if is_grounded and confidence >= self.confidence_threshold:
                logger.info(
                    f"Critic approved answer on iteration {iteration} "
                    f"(confidence={confidence:.2f})"
                )
                break
            else:
                logger.warning(
                    f"Critic rejected answer (grounded={is_grounded}, "
                    f"confidence={confidence:.2f}). Retrying..."
                )

        # If never approved, fall back to the best attempt
        if not (is_grounded and confidence >= self.confidence_threshold):
            logger.warning(
                f"Max iterations reached. Returning best answer "
                f"(confidence={best_confidence:.2f})."
            )
            answer = best_answer
            citations = best_citations
            citation_ids = best_citation_ids
            missing_aspects = best_missing
            hallucination_flags = best_flags
            confidence = best_confidence

        self.memory.add("assistant", answer)

        return AgentResponse(
            answer=answer,
            sources=attributed_sources,
            citations=citations,
            citation_ids=citation_ids,
            confidence=confidence,
            is_grounded=is_grounded,
            missing_aspects=missing_aspects,
            hallucination_flags=hallucination_flags,
            reasoning_trace=reasoning_trace,
            iterations=iteration,
            query=question,
        )
