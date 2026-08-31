#!/usr/bin/env python3
"""
Manual verification script for the multi-agent pipeline.

Loads the AgentExecutor and runs a sample question end-to-end,
printing the full reasoning trace so you can verify each agent's output.

Run from the research-assistant/ directory:
    python scripts/test_agent_pipeline.py
    python scripts/test_agent_pipeline.py --query "What is BERT?"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.orchestration.agent_executor import AgentExecutor

logger = get_logger("scripts.test_agent_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Test the full multi-agent pipeline.")
    parser.add_argument(
        "--query",
        type=str,
        default="What is the attention mechanism and how is it used in transformer models?",
        help="Research question to test the pipeline with.",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Number of docs to retrieve.")
    args = parser.parse_args()

    cfg = load_config()

    logger.info("Loading AgentExecutor (this will load the LLM)...")
    executor = AgentExecutor(cfg)

    logger.info(f"\nQuery: {args.query}\n")
    response = executor.query(args.query, top_k=args.top_k)

    print("\n" + "=" * 60)
    print("MULTI-AGENT PIPELINE RESULT")
    print("=" * 60)
    print(f"\nQuery: {response.query}\n")
    print(f"Answer:\n{response.answer}\n")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Grounded:   {response.is_grounded}")
    print(f"Iterations: {response.iterations}")
    print(f"\nCitations:")
    for c in response.citations:
        print(f"  - {c}")
    print(f"\nSources:")
    for s in response.sources:
        print(f"  - {s}")
    print("\n" + "-" * 60)
    print("REASONING TRACE:")
    print(json.dumps(response.reasoning_trace, indent=2))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
