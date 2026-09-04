# 🔬 Multi-Agent Research Assistant

A **production-grade multi-agent RAG system** that answers complex research questions by retrieving, reading, synthesizing, and critically validating answers from academic papers — all running locally on GPU with no external API calls.

---

## 📌 Project Status

| Phase | Description | Status | Key Result |
|-------|-------------|--------|------------|
| **Phase 1A** | Data Collection (500 ArXiv papers) | ✅ Done | 500 papers, ~40K chunks |
| **Phase 1B** | Synthetic Q&A Generation | ✅ Done | **5,000 Q&A pairs** (easy/medium/hard) |
| **Phase 1C** | Hybrid Retrieval (FAISS + BM25) | ✅ Done | **MRR@10: 0.8640** |
| **Phase 2 v1** | LoRA Fine-Tuning (1K data) | ✅ Done | Train: 1.346, Val: 1.556, gap: 0.21 |
| **Phase 2 v2** | LoRA Fine-Tuning (5K data, improved) | ✅ Done | **Train: 1.531, Val: 1.513, gap: 0.018** |
| **Phase 3** | Agent Robustness + Evaluation Suite | ✅ Done | **BERTScore: 0.7996, Hallucination: 1.5%** |
| **Phase 4** | API & Serving Layer (FastAPI) + Web UI | ✅ Done | **FastAPI, SSE Stream, Web Dashboard** |
| **Phase 5** | Containerization & Deployment (Docker) | 🔜 Next | — |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Hybrid Retriever              │
│   FAISS (semantic) + BM25 (keyword)     │
│   Weighted merge → Top-10 chunks        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│           Reader Agent                  │
│  Extracts the most relevant passages    │
│  from retrieved chunks                  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         Synthesizer Agent               │
│  Generates a cited answer strictly      │
│  from the extracted passages            │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│           Critic Agent                  │
│  Validates grounding & confidence.      │
│  Loops back to Synthesizer if needed.   │
└────────────────┬────────────────────────┘
                 │
                 ▼
         Final Answer + Citations
         + Full Reasoning Trace
```

---

## 🧠 Model

- **Base:** `mistralai/Mistral-7B-Instruct-v0.2`
- **Fine-tuning:** LoRA (`r=32, alpha=64`) via `trl` SFTTrainer — v2 improved config
- **Precision:** Native `bfloat16` (Blackwell GPU optimized, no quantization)
- **Training data:** **5,000 synthetic Q&A pairs** from 500 ArXiv papers (10 per paper)
- **Training time:** ~3.5 hours
- **Final train loss:** `1.531` | **Val loss:** `1.513` | **Gap: `0.018`** ← near-zero overfitting
- **Live test confidence:** `1.00` (grounded, zero hallucinations detected)
- **Adapter saved to:** `models/mistral-7b-finetuned-v2/`

---

## 📂 Project Structure

```
research-assistant/
├── config.yaml                  # Central configuration (all hyperparams here)
├── requirements.txt             # Python dependencies
├── src/
│   ├── data/                    # Data collection & processing
│   │   ├── arxiv_collector.py   # ArXiv paper downloader
│   │   ├── pdf_extractor.py     # PDF → text extraction
│   │   ├── text_chunker.py      # Token-aware chunking with overlap
│   │   └── qa_generator.py      # Synthetic Q&A generation via LLM
│   ├── retrieval/               # Hybrid retrieval system
│   │   ├── embeddings.py        # Sentence-transformer embeddings
│   │   ├── faiss_index.py       # FAISS vector index
│   │   ├── bm25_index.py        # BM25 keyword index
│   │   └── hybrid_retrieval.py  # Weighted score fusion
│   ├── models/
│   │   └── train.py             # LoRA SFT fine-tuning script
│   ├── agents/                  # Multi-agent system
│   │   ├── base.py              # Base agent + model loader (singleton)
│   │   ├── reader_agent.py      # Passage extraction agent
│   │   ├── synthesizer_agent.py # Answer synthesis with citations
│   │   └── critic_agent.py      # Hallucination detection & scoring
│   ├── api/                     # FastAPI application & web UI (Phase 4)
│   │   ├── app.py               # Lifespan manager, CORS, static mount
│   │   ├── routes.py            # /query, /stream (SSE), /health, /metrics
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── static/              # Interactive Web Dashboard (HTML/CSS/JS)
│   ├── evaluation/              # Quantitative benchmarking (Phase 3)
│   │   └── evaluator.py         # ROUGE, BERTScore, F1, Hallucination
│   ├── orchestration/
│   │   └── agent_executor.py    # 4-step pipeline orchestrator & SSE generator
│   ├── tools/
│   │   └── retrieval_tool.py    # Retrieval interface for agents
│   └── utils/
│       ├── config.py            # Typed config dataclasses
│       └── logger.py            # Structured logging
├── scripts/                     # Executable entry points
│   ├── collect_papers.py        # Run Phase 1A
│   ├── generate_qa.py           # Run Phase 1B
│   ├── build_index.py           # Run Phase 1C
│   ├── eval_retrieval.py        # Evaluate retrieval (MRR, Recall)
│   ├── run_finetuning.py        # Run Phase 2 fine-tuning
│   ├── test_agent_pipeline.py   # End-to-end pipeline test
│   ├── run_evaluation.py        # Run Phase 3 200-query benchmark
│   ├── serve_api.py             # Launch FastAPI server & Web UI (Phase 4)
│   └── run_ablations.py         # Run component ablation study (Phase 4)
├── data/                        # Raw & processed data (gitignored)
├── models/                      # Fine-tuned adapter weights (gitignored)
└── logs/                        # Training & evaluation logs (gitignored)
```

---

## 🚀 Quick Start

### 1. Setup

```bash
git clone <your-repo-url>
cd research-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Reproduce Phase 1 — Data & Retrieval

```bash
# Collect 500 papers from ArXiv (cs.LG, cs.CL, cs.AI)
python scripts/collect_papers.py

# Generate 5,000 Q&A pairs (requires GPU)
python scripts/generate_qa.py

# Build FAISS + BM25 indices
python scripts/build_index.py

# Evaluate retrieval baseline
python scripts/eval_retrieval.py
```

### 3. Reproduce Phase 2 — Fine-Tuning

```bash
# Train LoRA adapter on Mistral-7B
WANDB_MODE=offline python scripts/run_finetuning.py
```

### 4. Run Quantitative Benchmark (Phase 3)

```bash
# Evaluate 200 held-out test queries
python scripts/run_evaluation.py --num_samples 200
```

### 5. Launch API & Interactive Web Dashboard (Phase 4)

```bash
# Start FastAPI backend and serve web dashboard
python scripts/serve_api.py --port 8000

# Open in browser:
# • Interactive Web UI : http://localhost:8000/
# • OpenAPI Swagger Docs: http://localhost:8000/docs
# • Health Status Check : http://localhost:8000/health
```

---

## 📊 Phase 1 Results

| Metric | Score |
|--------|-------|
| MRR@10 (Hybrid) | **0.8640** |
| MRR@10 (FAISS only) | 0.8120 |
| MRR@10 (BM25 only) | 0.7340 |
| Recall@10 (Hybrid) | 0.9240 |

Hybrid retrieval (70% semantic + 30% keyword) outperforms either method alone.

## 📊 Phase 2 Results

### v1 → v2 Comparison

| Metric | v1 (1K data) | v2 (5K data) | Change |
|--------|-------------|-------------|--------|
| Training samples | 900 | **4,500** | +5x |
| Validation samples | 99 | **500** | +5x |
| LoRA rank (`r`) | 16 | **32** | doubled |
| LoRA dropout | 0.05 | **0.10** | stronger regularization |
| Learning rate | 2e-4 | **1e-4** | more stable |
| Epochs | 3 | **2** | more data = fewer passes |
| Trainable params | 13.6M (0.19%) | **27.3M (0.375%)** | more capacity |
| Final train loss | 1.346 | **1.531** | higher = less memorization |
| Final val loss | 1.556 | **1.513** | ✅ lower = better generalization |
| Train/val gap | 0.210 | **0.018** | ✅ 12x smaller, near-zero overfit |
| Pipeline confidence | 1.00 | **1.00** | maintained |
| Grounded | True | **True** | maintained |
| Iterations to approve | 1 | **1** | maintained |

---

## 📊 Phase 3 Evaluation Results (200 Held-Out Test Queries)

Evaluated end-to-end across 200 held-out academic queries with native GPU inference:

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Evaluation Completion** | **200 / 200 (0 errors)** | 200 | 100% stable |
| **Hallucination Rate** | **1.5%** | $\le 10\%$ | 🏆 Grounded |
| **BERTScore F1** | **0.7996** | $\ge 0.75$ | 🎯 Semantic alignment |
| **Average Confidence** | **0.998** | $\ge 0.85$ | 🌟 High certainty |
| **Average Iterations** | **1.04** | $\le 2.0$ | ⚡ 96% approved on pass 1 |
| **Token F1** | **0.3438** | — | High lexical precision |
| **ROUGE-L** | **0.2311** | — | Long-form cited explanations vs short references |
| **Average Pipeline Latency**| **76.6s / query** | — | 4-agent local GPU inference |

### Breakdown by Question Difficulty

| Difficulty | Queries | Token F1 | ROUGE-L | Hallucination Rate | Avg Confidence |
|------------|---------|----------|---------|--------------------|----------------|
| **Easy**   | 55      | 0.2116   | 0.1690  | 5.5%               | 0.991          |
| **Medium** | 78      | 0.4009   | 0.2640  | **0.0%**           | 1.000          |
| **Hard**   | 67      | 0.3857   | 0.2437  | **0.0%**           | 1.000          |

*Note: Medium and Hard queries achieved 0.0% hallucination with 1.00 confidence, demonstrating that the 3-strategy Reader Agent and hybrid retriever successfully capture complex context without degradation.*

---

## ⚙️ Configuration

All hyperparameters are centralized in [`config.yaml`](config.yaml). No hardcoded values in source code.

Key sections:
- `qa_generation` — controls dataset generation
- `finetuning` — LoRA rank, dropout, learning rate, epochs
- `retrieval` — FAISS/BM25 weights and top-K values
- `agents` — max iterations, confidence threshold

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| LLM | Mistral-7B-Instruct-v0.2 (HuggingFace) |
| Fine-tuning | `trl` (SFTTrainer) + `peft` (LoRA) |
| Vector search | `faiss-gpu` |
| Keyword search | `rank-bm25` |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Evaluation Metrics | `bert-score`, `rouge-score` |
| Training tracking | `wandb` (offline mode) |
| PDF extraction | `pdfminer.six` |
| Data source | ArXiv API |
| Serving & API | `fastapi`, `uvicorn`, `pydantic` |
| Web Dashboard | Vanilla HTML5, CSS3, JavaScript (ES6) |

---

## 🌐 Phase 4: API Serving & Interactive Web Dashboard

### Starting the Server
```bash
# Launch FastAPI server & interactive Web Dashboard on port 8000
python scripts/serve_api.py --port 8000
```

### Endpoints Overview
| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Interactive Single-Page Research Web Dashboard |
| `/docs` | `GET` | Interactive Swagger UI documentation with test console |
| `/openapi.json` | `GET` | Machine-readable OpenAPI 3.1.0 schema specification |
| `/api/v1/query` | `POST` | Synchronous 4-stage multi-agent inference |
| `/api/v1/stream` | `POST` / `GET` | Server-Sent Events (SSE) streaming real-time execution trace |
| `/health` | `GET` | System health, uptime, and Blackwell GPU VRAM occupancy |
| `/api/v1/metrics` | `GET` | Live telemetry (queries served, mean latency, groundedness %) |
| `/api/v1/history` | `GET` | In-memory rolling history of recent queries |

---

## 🔜 Upcoming (Phase 5)

- [ ] Docker containerization (`Dockerfile` & `docker-compose.yml`)
- [ ] Multi-hop query decomposition for comparative cross-paper synthesis
- [ ] Dual-backend provider (Gemini Pro API adapter for CPU-only execution)
- [ ] Production deployment & CI/CD pipeline

