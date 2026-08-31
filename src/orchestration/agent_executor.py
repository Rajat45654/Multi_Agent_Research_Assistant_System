"""
Agent Executor — Main Orchestration Loop.

This is the "brain coordinator" that manages all agents and tools.
It runs the full multi-agent pipeline in sequence:

    User Query
        → [RetrievalTool]      (search for relevant docs)
        → [ReaderAgent]        (extract relevant passages)
        → [SynthesizerAgent]   (write final answer)
        → [CriticAgent]        (validate, assign confidence)
        → If rejected: retry loop (max iterations)
        → Final Response

Design: Pure Python state machine — no LangChain overhead.
Complete traceability: every step is logged to a reasoning_trace list.
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
    sources: List[str]
    citations: List[str]
    confidence: float
    is_grounded: bool
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
    Orchestrates the multi-agent research pipeline.

    Usage:
        executor = AgentExecutor(cfg)
        response = executor.query("What is the transformer attention mechanism?")
        print(response.answer)
        print(response.reasoning_trace)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.max_iterations = cfg.agents.max_iterations
        self.confidence_threshold = cfg.agents.confidence_threshold

        logger.info("Initializing AgentExecutor...")

        # Load all tools and agents (model is loaded once via singleton in base.py)
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
            top_k: Number of documents to retrieve for context.

        Returns:
            An AgentResponse with the final answer and full reasoning trace.
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
        reader_output = self.reader.run(query=question, retrieved_docs=retrieved_docs)
        extracted_passages = reader_output["extracted_passages"]
        sources = reader_output["sources"]
        reasoning_trace.append({
            "step": "reader",
            "num_passages_extracted": len(extracted_passages),
            "passages_preview": [p[:100] for p in extracted_passages[:2]],
        })

        # ── Step 3 + 4: Synthesizer + Critic loop ─────────────────────────
        answer = ""
        citations = []
        confidence = 0.0
        is_grounded = False
        critic_feedback = ""
        iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"[Step 3/4] SynthesizerAgent: iteration {iteration}...")

            # Add critic feedback to passages context if retrying
            synthesis_context = extracted_passages
            if critic_feedback and iteration > 1:
                synthesis_context = [f"[Feedback from previous attempt: {critic_feedback}]"] + extracted_passages

            synth_output = self.synthesizer.run(
                query=question,
                extracted_passages=synthesis_context,
                sources=sources,
            )
            answer = synth_output["answer"]
            citations = synth_output["citations"]

            reasoning_trace.append({
                "step": f"synthesizer_iteration_{iteration}",
                "answer_preview": answer[:200],
                "num_citations": len(citations),
            })

            # Critic validation
            logger.info(f"[Step 4/4] CriticAgent: validating iteration {iteration}...")
            critic_output = self.critic.run(
                query=question,
                answer=answer,
                extracted_passages=extracted_passages,
            )
            is_grounded = critic_output["is_grounded"]
            confidence = critic_output["confidence"]
            critic_feedback = critic_output["feedback"]

            reasoning_trace.append({
                "step": f"critic_iteration_{iteration}",
                "is_grounded": is_grounded,
                "confidence": confidence,
                "feedback": critic_feedback,
            })

            # If grounded and confident enough, stop the loop
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

        self.memory.add("assistant", answer)

        return AgentResponse(
            answer=answer,
            sources=sources,
            citations=citations,
            confidence=confidence,
            is_grounded=is_grounded,
            reasoning_trace=reasoning_trace,
            iterations=iteration,
            query=question,
        )
