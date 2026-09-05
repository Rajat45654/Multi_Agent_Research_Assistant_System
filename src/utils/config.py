"""
Central configuration loader for the research assistant system.

Reads config.yaml and exposes typed dataclasses.
Supports environment variable overrides for server-side deployment.
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# ─────────────────────────────────────────────────────────────────
# Sub-configs (one per major section of config.yaml)
# ─────────────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    papers_dir: str = "data/raw/papers"
    processed_dir: str = "data/processed"
    embeddings_dir: str = "data/processed/embeddings"
    models_dir: str = "models"
    logs_dir: str = "logs"


@dataclass
class DataCollectionConfig:
    categories: List[str] = field(
        default_factory=lambda: ["cs.LG", "cs.CL", "cs.AI"]
    )
    num_papers: int = 500
    max_results_per_query: int = 100


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    overlap: int = 256
    tokenizer: str = "cl100k_base"


@dataclass
class EmbeddingsConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    device: str = "auto"


@dataclass
class RetrievalConfig:
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    top_k_semantic: int = 5
    top_k_bm25: int = 5
    top_k_final: int = 10


@dataclass
class QAGenerationConfig:
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    load_in_4bit: bool = True
    pairs_per_paper: int = 2
    target_total: int = 1000
    difficulty_split: Dict[str, float] = field(
        default_factory=lambda: {"easy": 0.40, "medium": 0.40, "hard": 0.20}
    )
    max_new_tokens: int = 256
    temperature: float = 0.7


@dataclass
class FineTuningConfig:
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    val_split: float = 0.1
    output_dir: str = "models/mistral-7b-finetuned"
    use_wandb: bool = True
    wandb_project: str = "research-assistant-finetune"
    save_steps: int = 50
    logging_steps: int = 10
    max_seq_length: int = 2048


@dataclass
class AgentConfig:
    max_iterations: int = 3
    confidence_threshold: float = 0.7
    memory_window: int = 10


@dataclass
class EvaluationConfig:
    test_set_size: int = 200
    bootstrap_iterations: int = 100
    metrics: List[str] = field(
        default_factory=lambda: [
            "exact_match", "f1", "bleu", "rouge_l", "citation_accuracy"
        ]
    )


@dataclass
class LLMConfig:
    backend: str = "local"                    # "local" (Mistral-7B GPU) or "gemini" (Cloud API, CPU)
    gemini_model: str = "gemini-3.6-flash"   # "gemini-3.6-flash", "gemini-2.5-pro", etc.
    gemini_api_key: str = ""



# ─────────────────────────────────────────────────────────────────
# Root config
# ─────────────────────────────────────────────────────────────────

@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    qa_generation: QAGenerationConfig = field(default_factory=QAGenerationConfig)
    finetuning: FineTuningConfig = field(default_factory=FineTuningConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from a YAML file with optional environment variable overrides.

    Env variable overrides (all optional):
        RA_PAPERS_DIR   — override paths.papers_dir
        RA_NUM_PAPERS   — override data_collection.num_papers
        RA_MODELS_DIR   — override paths.models_dir
        RA_LOGS_DIR     — override paths.logs_dir

    Args:
        config_path: Path to config.yaml (resolved relative to CWD).

    Returns:
        Fully populated :class:`Config` dataclass.

    Example::

        from src.utils.config import load_config
        cfg = load_config()
        print(cfg.paths.papers_dir)
    """
    config = Config()

    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        _apply_section(config, "paths", PathConfig, raw)
        _apply_section(config, "data_collection", DataCollectionConfig, raw)
        _apply_section(config, "chunking", ChunkingConfig, raw)
        _apply_section(config, "embeddings", EmbeddingsConfig, raw)
        _apply_section(config, "retrieval", RetrievalConfig, raw)
        _apply_section(config, "qa_generation", QAGenerationConfig, raw)
        _apply_section(config, "finetuning", FineTuningConfig, raw)
        _apply_section(config, "agents", AgentConfig, raw)
        _apply_section(config, "evaluation", EvaluationConfig, raw)
        _apply_section(config, "llm", LLMConfig, raw)

    # Automatically load .env file if present in workspace root
    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

    # Environment variable overrides
    if os.environ.get("RA_PAPERS_DIR"):
        config.paths.papers_dir = os.environ["RA_PAPERS_DIR"]
    if os.environ.get("RA_NUM_PAPERS"):
        config.data_collection.num_papers = int(os.environ["RA_NUM_PAPERS"])
    if os.environ.get("RA_MODELS_DIR"):
        config.paths.models_dir = os.environ["RA_MODELS_DIR"]
    if os.environ.get("RA_LOGS_DIR"):
        config.paths.logs_dir = os.environ["RA_LOGS_DIR"]

    # LLM Dual Backend overrides
    if os.environ.get("LLM_BACKEND"):
        config.llm.backend = os.environ["LLM_BACKEND"].strip().lower()
    if os.environ.get("GEMINI_API_KEY"):
        config.llm.gemini_api_key = os.environ["GEMINI_API_KEY"].strip()
    if os.environ.get("GEMINI_MODEL"):
        config.llm.gemini_model = os.environ["GEMINI_MODEL"].strip()

    return config


def _apply_section(
    config: Config,
    section: str,
    cls,
    raw: dict,
) -> None:
    """Apply a YAML section to the matching config dataclass."""
    if section in raw and isinstance(raw[section], dict):
        try:
            setattr(config, section, cls(**raw[section]))
        except TypeError as e:
            # Gracefully ignore unknown keys from future config additions
            known_fields = {f.name for f in cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in raw[section].items() if k in known_fields}
            setattr(config, section, cls(**filtered))
