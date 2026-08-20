# Multi-Agent Research Assistant

A **production-grade multi-agent RAG system** for answering complex multi-hop questions over research papers.

> Full README with architecture diagram, benchmark results, and API docs will be completed in Phase 5.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect papers (Phase 1A)
python scripts/collect_papers.py --num_papers 500

# 3. Generate Q&A pairs (Phase 1B — requires GPU)
python scripts/generate_qa.py

# 4. Build search indices (Phase 1C)
python scripts/build_index.py

# 5. Evaluate retrieval baseline
python scripts/eval_retrieval.py
```

## Project Structure

```
research-assistant/
├── config.yaml              # Central configuration
├── requirements.txt         # Python dependencies
├── src/
│   ├── data/                # Data collection & processing
│   ├── retrieval/           # FAISS + BM25 hybrid retrieval
│   ├── agents/              # Reader, Synthesizer, Critic agents (Phase 2)
│   ├── orchestration/       # Agent executor & memory (Phase 2)
│   ├── reasoning/           # Chain-of-thought, reflection (Phase 3)
│   ├── evaluation/          # Metrics & benchmarking (Phase 4)
│   ├── api/                 # FastAPI application (Phase 5)
│   └── utils/               # Logging, config
├── scripts/                 # Executable entry points
├── data/                    # Raw & processed data
├── models/                  # Fine-tuned model checkpoints
├── evals/                   # Evaluation datasets & results
└── deployment/              # Docker & deployment configs (Phase 5)
```

## Target Metrics

| Metric | Target |
|--------|--------|
| Exact Match | ≥ 68% |
| F1 Score | ≥ 0.72 |
| Cost/Query | ≤ $0.02 |
| Latency | ≤ 3s |
| Hallucination Rate | ≤ 10% |
