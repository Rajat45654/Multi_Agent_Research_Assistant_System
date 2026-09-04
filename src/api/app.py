"""
FastAPI application factory and lifespan manager for the Research Assistant.
"""

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.api.routes import router
from src.orchestration.agent_executor import AgentExecutor
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that handles startup initialization
    and clean shutdown of heavy resources (fine-tuned model & indices).
    """
    logger.info("Starting Multi-Agent Research Assistant API...")

    cfg = load_config()
    try:
        # Preload the fine-tuned model and hybrid search indices into app state
        executor = AgentExecutor(cfg)
        app.state.executor = executor
        app.state.config = cfg
        logger.info("AgentExecutor and models loaded successfully into application state.")
    except Exception as e:
        logger.error(f"Failed to initialize AgentExecutor on startup: {e}", exc_info=True)
        app.state.executor = None

    yield

    logger.info("Shutting down Research Assistant API...")
    app.state.executor = None


TAGS_METADATA = [
    {
        "name": "Inference",
        "description": "Multi-agent research and reasoning endpoints supporting both synchronous full synthesis and real-time Server-Sent Events (SSE) streaming.",
    },
    {
        "name": "Monitoring",
        "description": "Operational telemetry, GPU memory occupancy, service health checks, and rolling query history.",
    },
]


def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.
    """
    app = FastAPI(
        title="Multi-Agent Research Assistant API",
        description=(
            "### Multi-Agent Autonomous RAG System\n\n"
            "This interactive API provides access to a fine-tuned Mistral-7B multi-agent research assistant. "
            "It combines:\n"
            "- **Hybrid Document Retrieval**: Dense vector search (FAISS) + lexical ranking (BM25) with reciprocal rank fusion.\n"
            "- **Reader Agent**: Precise context extraction and quote verification.\n"
            "- **Synthesizer Agent**: Multi-hop cited answer generation with explicit `[Evidence N]` brackets.\n"
            "- **Critic Agent**: Hallucination detection, groundedness verification, and iterative refinement loop.\n\n"
            "**Web Interface**: Navigate to [`/`](/) to use the interactive visual reasoning dashboard."
        ),
        version="1.0.0",
        openapi_tags=TAGS_METADATA,
        swagger_ui_parameters={
            "defaultModelsExpandDepth": 2,
            "docExpansion": "list",
            "filter": True,
            "showExtensions": True,
        },
        lifespan=lifespan,
    )

    # Enable CORS for local development and web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include REST API endpoints
    app.include_router(router)

    # Mount static assets for web UI
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return {"message": "Multi-Agent Research Assistant API is running. Visit /docs for OpenAPI specs."}

    return app


# Application instance for uvicorn entry
app = create_app()
