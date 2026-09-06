"""
Unit tests for the Academic Exporter (Phase 5).
"""

import unittest
from src.utils.exporter import generate_bibtex, generate_markdown_report


class TestExporter(unittest.TestCase):
    def test_generate_bibtex(self):
        sources = ["arXiv:1706.03762", "arXiv:2301.00001", "unknown"]
        bib = generate_bibtex(sources)
        self.assertIn("@article{arxiv_1706_03762", bib)
        self.assertIn("@article{arxiv_2301_00001", bib)
        self.assertIn("https://arxiv.org/abs/1706.03762", bib)
        self.assertNotIn("unknown", bib)

    def test_generate_bibtex_empty(self):
        bib = generate_bibtex([])
        self.assertIn("@misc{research_assistant_report", bib)

    def test_generate_markdown_report(self):
        report = generate_markdown_report(
            query="What is attention in transformers?",
            answer="Attention is a mechanism [Evidence 1].",
            confidence=0.98,
            is_grounded=True,
            sources=["arXiv:1706.03762"],
            citations=["[Evidence 1]"],
            reasoning_trace=[{"step": "retrieval", "count": 10}],
        )
        self.assertIn("# 📄 Academic Research Report", report)
        self.assertIn("What is attention in transformers?", report)
        self.assertIn("0.98", report)
        self.assertIn("https://arxiv.org/abs/1706.03762", report)
        self.assertIn("Multi-Agent Pipeline Trace", report)


if __name__ == "__main__":
    unittest.main()
