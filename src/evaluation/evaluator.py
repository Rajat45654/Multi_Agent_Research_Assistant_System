"""
Evaluator — Phase 3.

Runs the multi-agent pipeline over a held-out test set and computes:
  - ROUGE-L          (n-gram overlap)
  - BERTScore F1     (semantic similarity)
  - Token-level F1   (same as QA eval F1)
  - Exact Match      (strict string match after normalization)
  - Citation Accuracy (fraction of cited evidence IDs that were retrieved)
  - Hallucination Rate (fraction of answers the Critic marks as not grounded)
  - Avg Confidence   (mean critic confidence across all queries)
  - Avg Iterations   (mean refinement loops needed)
  - Avg Latency      (seconds per query)
"""

import json
import re
import time
import string
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Text normalization helpers (standard QA normalization)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator class
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Evaluates the full multi-agent pipeline on a held-out test set.

    Usage:
        from src.orchestration.agent_executor import AgentExecutor
        evaluator = Evaluator(executor, output_dir="evals/results")
        results = evaluator.run(qa_pairs, num_samples=200)
        evaluator.save(results)
    """

    def __init__(self, executor, output_dir: str = "evals/results"):
        self.executor = executor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-load heavy metrics
        self._rouge = None
        self._bertscore = None

    def _get_rouge(self):
        if self._rouge is None:
            from rouge_score import rouge_scorer
            self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return self._rouge

    def _get_bertscore(self):
        if self._bertscore is None:
            from bert_score import score as bert_score_fn
            self._bertscore = bert_score_fn
        return self._bertscore

    def _compute_rouge_l(self, pred: str, gold: str) -> float:
        scorer = self._get_rouge()
        scores = scorer.score(gold, pred)
        return scores["rougeL"].fmeasure

    def _compute_bertscore(self, preds: List[str], golds: List[str]) -> List[float]:
        bert_score_fn = self._get_bertscore()
        # Run in batch for efficiency
        P, R, F = bert_score_fn(
            preds, golds,
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
        )
        return F.tolist()

    def evaluate_single(self, qa_pair: dict) -> dict:
        """Run pipeline on one QA pair and return per-example metrics."""
        question = qa_pair["question"]
        reference = qa_pair["ground_truth_answer"]

        start = time.time()
        try:
            response = self.executor.query(question)
            latency = time.time() - start

            pred_answer = response.answer
            return {
                "question": question,
                "reference": reference,
                "prediction": pred_answer,
                "confidence": response.confidence,
                "is_grounded": response.is_grounded,
                "iterations": response.iterations,
                "citation_ids": response.citation_ids,
                "missing_aspects": response.missing_aspects,
                "hallucination_flags": response.hallucination_flags,
                "sources": response.sources,
                "latency": latency,
                "difficulty": qa_pair.get("difficulty_level", "unknown"),
                # Per-example metrics computed later in batch
                "f1": _token_f1(pred_answer, reference),
                "exact_match": _exact_match(pred_answer, reference),
                "error": None,
            }
        except Exception as e:
            latency = time.time() - start
            logger.error(f"Pipeline error for query '{question[:60]}': {e}")
            return {
                "question": question,
                "reference": reference,
                "prediction": "",
                "confidence": 0.0,
                "is_grounded": False,
                "iterations": 0,
                "citation_ids": [],
                "missing_aspects": [],
                "hallucination_flags": [],
                "sources": [],
                "latency": latency,
                "difficulty": qa_pair.get("difficulty_level", "unknown"),
                "f1": 0.0,
                "exact_match": False,
                "error": str(e),
            }

    def run(self, qa_pairs: List[dict], num_samples: int = 200) -> dict:
        """
        Run evaluation on the last `num_samples` pairs (held-out test set).

        Args:
            qa_pairs   : Full list of QA pairs from qa_pairs.json
            num_samples: How many to evaluate (taken from end of list)

        Returns:
            dict with per-example results and aggregate metrics
        """
        # Use last N pairs as test set (never seen during training)
        test_pairs = qa_pairs[-num_samples:]
        logger.info(f"Running evaluation on {len(test_pairs)} test pairs...")

        per_example = []
        for i, pair in enumerate(test_pairs):
            logger.info(f"  [{i+1}/{len(test_pairs)}] {pair['question'][:60]}...")
            result = self.evaluate_single(pair)
            per_example.append(result)

            # Print running summary every 10 examples
            if (i + 1) % 10 == 0:
                done = per_example
                avg_f1 = sum(r["f1"] for r in done) / len(done)
                avg_conf = sum(r["confidence"] for r in done) / len(done)
                hall_rate = sum(1 for r in done if not r["is_grounded"]) / len(done)
                logger.info(
                    f"  Progress {i+1}/{len(test_pairs)} | "
                    f"F1={avg_f1:.3f} | Conf={avg_conf:.3f} | "
                    f"Hall%={hall_rate*100:.1f}%"
                )

        # Compute ROUGE-L and BERTScore in batch
        logger.info("Computing ROUGE-L scores...")
        for r in per_example:
            if r["prediction"]:
                r["rouge_l"] = self._compute_rouge_l(r["prediction"], r["reference"])
            else:
                r["rouge_l"] = 0.0

        logger.info("Computing BERTScore (batch)...")
        valid = [(i, r) for i, r in enumerate(per_example) if r["prediction"]]
        if valid:
            idxs, valid_results = zip(*valid)
            preds = [r["prediction"] for r in valid_results]
            golds = [r["reference"] for r in valid_results]
            bert_scores = self._compute_bertscore(preds, golds)
            for idx, bs in zip(idxs, bert_scores):
                per_example[idx]["bertscore_f1"] = bs
        for r in per_example:
            if "bertscore_f1" not in r:
                r["bertscore_f1"] = 0.0

        # Aggregate metrics
        n = len(per_example)
        n_valid = sum(1 for r in per_example if not r["error"])

        aggregate = {
            "num_evaluated": n,
            "num_valid": n_valid,
            "num_errors": n - n_valid,
            "rouge_l": sum(r["rouge_l"] for r in per_example) / n,
            "bertscore_f1": sum(r["bertscore_f1"] for r in per_example) / n,
            "token_f1": sum(r["f1"] for r in per_example) / n,
            "exact_match": sum(r["exact_match"] for r in per_example) / n,
            "hallucination_rate": sum(1 for r in per_example if not r["is_grounded"]) / n,
            "avg_confidence": sum(r["confidence"] for r in per_example) / n,
            "avg_iterations": sum(r["iterations"] for r in per_example) / n,
            "avg_latency_s": sum(r["latency"] for r in per_example) / n,
            # Breakdown by difficulty
            "by_difficulty": self._breakdown_by_difficulty(per_example),
        }

        logger.info("Evaluation complete.")
        return {"aggregate": aggregate, "per_example": per_example}

    def _breakdown_by_difficulty(self, results: List[dict]) -> dict:
        breakdown = {}
        for diff in ["easy", "medium", "hard"]:
            subset = [r for r in results if r.get("difficulty") == diff]
            if not subset:
                continue
            n = len(subset)
            breakdown[diff] = {
                "n": n,
                "rouge_l": sum(r.get("rouge_l", 0) for r in subset) / n,
                "token_f1": sum(r["f1"] for r in subset) / n,
                "hallucination_rate": sum(1 for r in subset if not r["is_grounded"]) / n,
                "avg_confidence": sum(r["confidence"] for r in subset) / n,
            }
        return breakdown

    def save(self, results: dict, tag: str = "phase3") -> Path:
        """Save full results JSON and a human-readable summary markdown."""
        json_path = self.output_dir / f"{tag}_eval.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Full results saved to {json_path}")

        summary_path = self.output_dir / f"{tag}_summary.md"
        self._write_summary(results["aggregate"], summary_path)
        logger.info(f"Summary saved to {summary_path}")

        return json_path

    def _write_summary(self, agg: dict, path: Path):
        lines = [
            "# Phase 3 Evaluation Summary\n",
            f"| Metric | Score |",
            f"|--------|-------|",
            f"| Test queries | {agg['num_evaluated']} ({agg['num_errors']} errors) |",
            f"| ROUGE-L | {agg['rouge_l']:.4f} |",
            f"| BERTScore F1 | {agg['bertscore_f1']:.4f} |",
            f"| Token F1 | {agg['token_f1']:.4f} |",
            f"| Exact Match | {agg['exact_match']*100:.1f}% |",
            f"| Hallucination Rate | {agg['hallucination_rate']*100:.1f}% |",
            f"| Avg Confidence | {agg['avg_confidence']:.3f} |",
            f"| Avg Iterations | {agg['avg_iterations']:.2f} |",
            f"| Avg Latency | {agg['avg_latency_s']:.1f}s/query |",
            "",
            "## By Difficulty",
            "",
        ]
        for diff, stats in agg.get("by_difficulty", {}).items():
            lines += [
                f"### {diff.capitalize()} (n={stats['n']})",
                f"- ROUGE-L: {stats['rouge_l']:.4f}",
                f"- Token F1: {stats['token_f1']:.4f}",
                f"- Hallucination: {stats['hallucination_rate']*100:.1f}%",
                f"- Avg Confidence: {stats['avg_confidence']:.3f}",
                "",
            ]
        path.write_text("\n".join(lines))
