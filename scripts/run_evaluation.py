"""
run_evaluation.py — Phase 3 Evaluation Script.

Runs the full multi-agent pipeline over 200 held-out test queries and
computes ROUGE-L, BERTScore, Token-F1, Exact Match, Hallucination Rate,
Avg Confidence, Avg Iterations, and Avg Latency.

Usage:
    python scripts/run_evaluation.py                    # 200 test queries (default)
    python scripts/run_evaluation.py --num_samples 50   # quick 50-query smoke test
    python scripts/run_evaluation.py --tag phase3_v2    # custom output tag

Monitor progress:
    tail -f logs/evaluation.log
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.orchestration.agent_executor import AgentExecutor
from src.evaluation.evaluator import Evaluator

logger = get_logger("scripts.run_evaluation")


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 3 evaluation on the held-out test set."
    )
    parser.add_argument(
        "--num_samples", type=int, default=200,
        help="Number of test queries to evaluate (default: 200)"
    )
    parser.add_argument(
        "--tag", type=str, default="phase3",
        help="Output file tag (default: phase3)"
    )
    parser.add_argument(
        "--qa_file", type=str, default="data/processed/qa_pairs.json",
        help="Path to QA pairs JSON file"
    )
    args = parser.parse_args()

    logger.info("=== Phase 3 Evaluation Starting ===")
    logger.info(f"  Test samples : {args.num_samples}")
    logger.info(f"  QA file      : {args.qa_file}")
    logger.info(f"  Output tag   : {args.tag}")

    # Load QA pairs
    qa_path = Path(args.qa_file)
    if not qa_path.exists():
        logger.error(f"QA pairs file not found: {qa_path}")
        sys.exit(1)

    with open(qa_path) as f:
        qa_pairs = json.load(f)

    logger.info(f"Loaded {len(qa_pairs)} total QA pairs.")
    logger.info(f"Using last {args.num_samples} as held-out test set.")

    # Initialize pipeline
    cfg = load_config()
    executor = AgentExecutor(cfg)

    # Run evaluation
    evaluator = Evaluator(executor, output_dir="evals/results")
    results = evaluator.run(qa_pairs, num_samples=args.num_samples)

    # Save results
    out_path = evaluator.save(results, tag=args.tag)

    # Print summary to terminal
    agg = results["aggregate"]
    print("\n" + "=" * 55)
    print("  PHASE 3 EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Test queries      : {agg['num_evaluated']} ({agg['num_errors']} errors)")
    print(f"  ROUGE-L           : {agg['rouge_l']:.4f}")
    print(f"  BERTScore F1      : {agg['bertscore_f1']:.4f}")
    print(f"  Token F1          : {agg['token_f1']:.4f}")
    print(f"  Exact Match       : {agg['exact_match']*100:.1f}%")
    print(f"  Hallucination Rate: {agg['hallucination_rate']*100:.1f}%")
    print(f"  Avg Confidence    : {agg['avg_confidence']:.3f}")
    print(f"  Avg Iterations    : {agg['avg_iterations']:.2f}")
    print(f"  Avg Latency       : {agg['avg_latency_s']:.1f}s/query")
    print("=" * 55)
    print(f"\nFull results: {out_path}")
    print(f"Summary:      evals/results/{args.tag}_summary.md\n")

    logger.info("=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
