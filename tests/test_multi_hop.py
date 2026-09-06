"""
Unit tests for QueryDecomposer and multi-hop retrieval fusion (Phase 5).
"""

import unittest
from unittest.mock import MagicMock
from src.orchestration.query_decomposer import QueryDecomposer


class TestQueryDecomposer(unittest.TestCase):
    def setUp(self):
        self.mock_cfg = MagicMock()
        self.decomposer = QueryDecomposer(self.mock_cfg)

    def test_comparative_detection(self):
        comparative_queries = [
            "Compare LoRA and QLoRA in memory efficiency",
            "What is the difference between FlashAttention and self-attention?",
            "Contrast sparse attention versus dense attention",
            "Trade-offs between quantized models and full precision",
            "BERT vs GPT architectures in NLP",
            "How does AdamW differ from standard Adam?",
        ]
        for q in comparative_queries:
            self.assertTrue(self.decomposer.is_multi_hop(q), f"Failed to detect comparative query: {q}")

    def test_single_hop_passthrough(self):
        single_hop_queries = [
            "What is the attention mechanism in transformers?",
            "Explain rotary positional encodings",
            "How does backpropagation work in neural networks?",
        ]
        for q in single_hop_queries:
            self.assertFalse(self.decomposer.is_multi_hop(q), f"Incorrectly flagged atomic query: {q}")
            self.assertEqual(self.decomposer.decompose(q), [q])

    def test_decomposition_splitting(self):
        q = "Compare LoRA and QLoRA in terms of memory efficiency"
        sub = self.decomposer.decompose(q)
        self.assertEqual(len(sub), 2)
        self.assertTrue(any("lora" in s.lower() for s in sub))
        self.assertTrue(any("qlora" in s.lower() for s in sub))

    def test_reciprocal_rank_fusion(self):
        list_a = [
            {"chunk_id": "c1", "arxiv_id": "2301.0001", "text": "Paper 1 chunk"},
            {"chunk_id": "c2", "arxiv_id": "2301.0002", "text": "Paper 2 chunk"},
        ]
        list_b = [
            {"chunk_id": "c3", "arxiv_id": "2301.0003", "text": "Paper 3 chunk"},
            {"chunk_id": "c1", "arxiv_id": "2301.0001", "text": "Paper 1 chunk"},
        ]
        fused = self.decomposer.fuse_results([list_a, list_b], top_k=3)
        self.assertEqual(len(fused), 3)
        # c1 appeared in both lists, so it must rank #1
        self.assertEqual(fused[0]["chunk_id"], "c1")


if __name__ == "__main__":
    unittest.main()
