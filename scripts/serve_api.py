"""
CLI Launcher for the Multi-Agent Research Assistant FastAPI server.

Usage:
    python scripts/serve_api.py --port 8080
    python scripts/serve_api.py --port 8080 --host 0.0.0.0
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from src.utils.logger import get_logger

logger = get_logger("scripts.serve_api")

def main():
    import os
    default_port = int(os.environ.get("PORT", 8080))
    parser = argparse.ArgumentParser(description="Start the Multi-Agent Research Assistant API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to bind (default: {default_port})")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for local development")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["local", "gemini"],
        default=None,
        help="Override LLM backend: 'local' (Mistral-7B on GPU) or 'gemini' (Gemini Cloud API, CPU only)"
    )
    args = parser.parse_args()

    import os
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend

    backend_active = args.backend or os.environ.get("LLM_BACKEND", "local")

    print("\n" + "=" * 60)
    print("  🔬 MULTI-AGENT RESEARCH ASSISTANT API SERVER")
    print("=" * 60)
    print(f"  • LLM Backend   : {backend_active.upper()} {'(Cloud API / CPU-only)' if backend_active == 'gemini' else '(Mistral-7B / GPU)'}")
    print(f"  • Web Dashboard : http://localhost:{args.port}/")
    print(f"  • Swagger Docs  : http://localhost:{args.port}/docs")
    print(f"  • OpenAPI Spec  : http://localhost:{args.port}/openapi.json")
    print(f"  • Health Check  : http://localhost:{args.port}/health")
    print("=" * 60 + "\n")

    # Run uvicorn server (1 worker owns the GPU VRAM cleanly)
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
