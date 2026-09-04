"""
FastAPI route definitions for the Multi-Agent Research Assistant API.
"""

import time
import json
import asyncio
from datetime import datetime
from typing import List, Optional

import torch
from fastapi import APIRouter, HTTPException, Request, Depends, Query
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    MetricsResponse,
    HistoryItem,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# In-memory telemetry and query session history
START_TIME = time.time()
TOTAL_QUERIES = 0
TOTAL_ERRORS = 0
TOTAL_LATENCY = 0.0
TOTAL_CONFIDENCE = 0.0
TOTAL_GROUNDED = 0
QUERY_HISTORY: List[HistoryItem] = []

# Concurrency guard: prevents multiple simultaneous requests from saturating GPU VRAM
gpu_lock = asyncio.Lock()


def get_executor(request: Request):
    """Dependency helper to retrieve the preloaded AgentExecutor from app state."""
    executor = getattr(request.app.state, "executor", None)
    if executor is None:
        raise HTTPException(
            status_code=503,
            detail="AgentExecutor is not initialized. Check server logs."
        )
    return executor


# ─────────────────────────────────────────────────────────────────────────────
# Health & Status Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check(request: Request):
    """
    Health check endpoint returning system status, uptime, and GPU memory usage.
    """
    gpu_name = None
    gpu_used_mb = None
    gpu_total_mb = None

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_used_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
        gpu_total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2)

    executor = getattr(request.app.state, "executor", None)
    model_loaded = executor is not None and getattr(executor, "reader", None) is not None
    indices_loaded = executor is not None and getattr(executor, "retrieval_tool", None) is not None

    return HealthResponse(
        status="healthy" if (model_loaded and indices_loaded) else "degraded",
        gpu_device_name=gpu_name,
        gpu_memory_used_mb=gpu_used_mb,
        gpu_memory_total_mb=gpu_total_mb,
        model_loaded=model_loaded,
        indices_loaded=indices_loaded,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@router.get("/api/v1/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Serving metrics covering query volume, mean latency, and hallucination rates.
    """
    avg_latency = (TOTAL_LATENCY / TOTAL_QUERIES) if TOTAL_QUERIES > 0 else 0.0
    avg_conf = (TOTAL_CONFIDENCE / TOTAL_QUERIES) if TOTAL_QUERIES > 0 else 0.0
    grounded_pct = (TOTAL_GROUNDED / TOTAL_QUERIES * 100.0) if TOTAL_QUERIES > 0 else 100.0

    return MetricsResponse(
        total_queries_served=TOTAL_QUERIES,
        avg_latency_seconds=round(avg_latency, 2),
        avg_confidence=round(avg_conf, 3),
        grounded_rate_percent=round(grounded_pct, 1),
        total_errors=TOTAL_ERRORS,
    )


@router.get("/api/v1/history", response_model=List[HistoryItem], tags=["Monitoring"])
async def get_history():
    """
    Returns rolling in-memory history of the last 20 queries.
    """
    return QUERY_HISTORY[-20:]


# ─────────────────────────────────────────────────────────────────────────────
# Synchronous Query Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    tags=["Inference"],
    summary="Query Assistant (Synchronous)",
    responses={
        200: {
            "model": QueryResponse,
            "description": "Full multi-agent synthesized response with grounded citations and execution trace."
        },
        500: {"description": "Internal server / inference error during agent execution."},
        503: {"description": "AgentExecutor or model not initialized."},
    },
)
async def query_assistant(req: QueryRequest, request: Request):
    """
    Executes the full 4-stage multi-agent pipeline (Retrieval -> Reader -> Synthesizer -> Critic)
    and returns a structured response.

    **Note**: Deep autoregressive generation across multi-agent validation passes
    takes approximately 60–90 seconds on GPU. For interactive real-time updates,
    use the streaming endpoints (`/api/v1/stream`).
    """
    global TOTAL_QUERIES, TOTAL_ERRORS, TOTAL_LATENCY, TOTAL_CONFIDENCE, TOTAL_GROUNDED

    executor = get_executor(request)
    start_time = time.time()

    async with gpu_lock:
        try:
            # Run heavy agent pipeline in worker thread to avoid blocking FastAPI event loop
            response = await asyncio.to_thread(
                executor.query,
                question=req.query,
                top_k=req.top_k,
            )
        except Exception as e:
            TOTAL_ERRORS += 1
            logger.error(f"Error during query execution: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    latency = round(time.time() - start_time, 2)

    # Update telemetry
    TOTAL_QUERIES += 1
    TOTAL_LATENCY += latency
    TOTAL_CONFIDENCE += response.confidence
    if response.is_grounded:
        TOTAL_GROUNDED += 1

    history_entry = HistoryItem(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        query=req.query,
        answer_snippet=response.answer[:150] + ("..." if len(response.answer) > 150 else ""),
        confidence=response.confidence,
        is_grounded=response.is_grounded,
        latency_seconds=latency,
    )
    QUERY_HISTORY.append(history_entry)

    return QueryResponse(
        query=response.query,
        answer=response.answer,
        sources=response.sources,
        citations=response.citations,
        citation_ids=response.citation_ids,
        evidence=response.evidence,
        confidence=response.confidence,
        is_grounded=response.is_grounded,
        missing_aspects=response.missing_aspects,
        hallucination_flags=response.hallucination_flags,
        reasoning_trace=response.reasoning_trace,
        iterations=response.iterations,
        latency_seconds=latency,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Server-Sent Events (SSE) Endpoint
# ─────────────────────────────────────────────────────────────────────────────

async def event_generator(executor, query: str, top_k: int, max_iterations: Optional[int]):
    """Yields formatted SSE strings as the multi-agent system progresses."""
    global TOTAL_QUERIES, TOTAL_ERRORS, TOTAL_LATENCY, TOTAL_CONFIDENCE, TOTAL_GROUNDED
    start_time = time.time()

    async with gpu_lock:
        try:
            # We iterate over the stream generator
            # Use an iterator queue or run in executor
            loop = asyncio.get_running_loop()

            def run_stream():
                return list(executor.stream_query(
                    question=query,
                    top_k=top_k,
                    max_iterations=max_iterations,
                ))

            # Execute stream and emit events
            events = await loop.run_in_executor(None, run_stream)
            for ev in events:
                step_name = ev.get("step", "update")
                payload = json.dumps(ev)
                yield f"event: {step_name}\ndata: {payload}\n\n"
                await asyncio.sleep(0.01)

            # Record telemetry from final step
            latency = round(time.time() - start_time, 2)
            TOTAL_QUERIES += 1
            TOTAL_LATENCY += latency

            final_event = events[-1] if events else None
            if final_event and final_event.get("step") == "complete":
                data = final_event.get("data", {})
                TOTAL_CONFIDENCE += data.get("confidence", 1.0)
                if data.get("is_grounded", True):
                    TOTAL_GROUNDED += 1

                QUERY_HISTORY.append(HistoryItem(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    query=query,
                    answer_snippet=data.get("answer", "")[:150] + "...",
                    confidence=data.get("confidence", 1.0),
                    is_grounded=data.get("is_grounded", True),
                    latency_seconds=latency,
                ))

        except Exception as e:
            TOTAL_ERRORS += 1
            logger.error(f"Streaming error: {e}", exc_info=True)
            err_payload = json.dumps({"step": "error", "message": str(e), "data": {}})
            yield f"event: error\ndata: {err_payload}\n\n"


@router.post(
    "/api/v1/stream",
    tags=["Inference"],
    summary="Stream Query Events (POST)",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "example": "event: retrieval\ndata: {\"step\": \"retrieval\", \"message\": \"Retrieved 8 document chunks from FAISS + BM25.\", \"data\": {...}}\n\n"
                    }
                }
            },
            "description": "Continuous Server-Sent Events (SSE) stream emitting real-time agent execution events."
        },
        422: {"description": "Validation Error"},
        503: {"description": "AgentExecutor or model not initialized."}
    }
)
async def stream_query_post(req: QueryRequest, request: Request):
    """
    Streams multi-agent reasoning events in real time via Server-Sent Events (SSE).
    Emits events for each pipeline step: `retrieval`, `reader`, `synthesizer`, `critic`, and `complete`.
    """
    executor = get_executor(request)
    return StreamingResponse(
        event_generator(
            executor=executor,
            query=req.query,
            top_k=req.top_k,
            max_iterations=req.max_iterations,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get(
    "/api/v1/stream",
    tags=["Inference"],
    summary="Stream Query Events (GET)",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "example": "event: retrieval\ndata: {\"step\": \"retrieval\", \"message\": \"Retrieved 8 document chunks from FAISS + BM25.\", \"data\": {...}}\n\n"
                    }
                }
            },
            "description": "Continuous Server-Sent Events (SSE) stream for native EventSource browser connections."
        },
        422: {"description": "Validation Error"},
        503: {"description": "AgentExecutor or model not initialized."}
    }
)
async def stream_query_get(
    request: Request,
    query: str = Query(
        ...,
        min_length=3,
        max_length=1500,
        description="The research question to be answered by the multi-agent system.",
        example="What is attention in transformers?"
    ),
    top_k: int = Query(
        10,
        ge=1,
        le=30,
        description="Number of document chunks to retrieve.",
        example=10
    ),
    max_iterations: Optional[int] = Query(
        None,
        ge=1,
        le=5,
        description="Optional override for maximum Critic-Synthesizer refinement iterations.",
        example=3
    ),
):
    """
    GET variant for native browser `EventSource` connections streaming SSE events.
    """
    executor = get_executor(request)
    return StreamingResponse(
        event_generator(
            executor=executor,
            query=query,
            top_k=top_k,
            max_iterations=max_iterations,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
