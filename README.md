# 🔬 Multi-Agent Research Assistant

A **production-grade multi-agent RAG system** that answers complex research questions by retrieving, reading, synthesizing, and critically validating answers from academic papers — all running locally on GPU with no external API calls.

---

## 📌 Project Status

| Phase | Description | Status | Key Result |
|-------|-------------|--------|------------|
| **Phase 1A** | Data Collection (500 ArXiv papers) | ✅ Done | 500 papers, ~40K chunks |
| **Phase 1B** | Synthetic Q&A Generation | ✅ Done | 999 Q&A pairs (easy/medium/hard) |
| **Phase 1C** | Hybrid Retrieval (FAISS + BM25) | ✅ Done | **MRR@10: 0.8640** |
| **Phase 2** | LoRA Fine-Tuning + Multi-Agent Pipeline | ✅ Done | **Train loss: 1.346, Confidence: 1.00** |
| **Phase 3** | Iterative Refinement + Evaluation | 🔜 Planned | — |
| **Phase 4** | Benchmarking & Metrics | 🔜 Planned | — |
| **Phase 5** | API + Deployment | 🔜 Planned | — |

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
- **Fine-tuning:** LoRA (`r=16, alpha=32`) via `trl` SFTTrainer
- **Precision:** Native `bfloat16` (Blackwell GPU optimized, no quantization)
- **Training data:** 999 synthetic Q&A pairs from 500 ArXiv papers
- **Training time:** ~75 minutes
- **Final train loss:** `1.346` | **Val loss:** `1.556`
- **Live test confidence:** `1.00` (grounded, zero hallucinations detected)

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
│   ├── orchestration/
│   │   └── agent_executor.py    # 4-step pipeline orchestrator
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
│   └── test_agent_pipeline.py   # End-to-end pipeline test
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

# Generate 1,000 Q&A pairs (requires GPU, ~2-3 hrs)
python scripts/generate_qa.py 2>&1 | tee logs/qa_generation.log

# Build FAISS + BM25 indices
python scripts/build_index.py

# Evaluate retrieval baseline
python scripts/eval_retrieval.py
# Expected: MRR@10 ≈ 0.8640
```

### 3. Reproduce Phase 2 — Fine-Tuning

```bash
# Train LoRA adapter on Mistral-7B (~75 min on 2×GPU)
WANDB_MODE=offline python scripts/run_finetuning.py 2>&1 | tee logs/finetuning.log
# Expected: train_loss ≈ 1.346, val_loss ≈ 1.556
```

### 4. Run the Full Agent Pipeline

```bash
python scripts/test_agent_pipeline.py --query "What is attention in transformers?"
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

| Metric | Value |
|--------|-------|
| Training samples | 900 (90% of 999) |
| Validation samples | 99 (10% of 999) |
| LoRA trainable params | 13.6M / 7.26B (0.19%) |
| Training duration | ~75 min |
| Final train loss | 1.346 |
| Final val loss | 1.556 |
| Pipeline confidence score | **1.00** |
| Grounded (no hallucination) | **True** |
| Iterations to approve | 1 |

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
| Training tracking | `wandb` (offline mode) |
| PDF extraction | `pdfminer.six` |
| Data source | ArXiv API |

---

## 🔜 Upcoming (Phase 3+)

- [ ] Scale data to 5,000 Q&A pairs for better generalization
- [ ] Formal evaluation suite (ROUGE, BERTScore, F1, EM)
- [ ] Iterative Critic→Synthesizer refinement loop
- [ ] FastAPI serving layer
- [ ] Docker deployment
