"""
Base Agent class that all agents inherit from.

Provides a standard interface for loading the fine-tuned model
and generating responses via prompt templates.
"""

import torch
from pathlib import Path
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)

# Shared model instance (loaded once, reused by all agents)
_shared_model = None
_shared_tokenizer = None


def load_model(cfg: Config):
    """
    Loads the fine-tuned model (LoRA adapter merged on top of base).
    Uses a module-level singleton so all agents share one copy in VRAM.
    Falls back to the base model if no fine-tuned checkpoint is found.
    """
    global _shared_model, _shared_tokenizer

    if _shared_model is not None:
        return _shared_model, _shared_tokenizer

    finetuned_dir = Path(cfg.finetuning.output_dir)  # models/mistral-7b-finetuned-v2
    # Also check for v1 as fallback
    finetuned_dir_v1 = Path(str(finetuned_dir).replace("-v2", ""))
    base_model_name = cfg.finetuning.base_model

    logger.info("Loading model for agent inference...")

    # Determine which adapter to load (prefer v2)
    adapter_dir = None
    for candidate in [finetuned_dir, finetuned_dir_v1]:
        if candidate.exists() and (candidate / "adapter_config.json").exists():
            adapter_dir = candidate
            break

    tok_dir = str(adapter_dir) if adapter_dir else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_dir:
        logger.info(f"Loading LoRA adapter from {adapter_dir}")
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        model = model.merge_and_unload()  # Merge LoRA weights for fast inference
        logger.info("Fine-tuned model loaded and merged.")
    else:
        logger.warning(
            f"No fine-tuned adapter found. "
            "Using base model. Run scripts/run_finetuning.py first."
        )
        model = base

    model.eval()
    _shared_model = model
    _shared_tokenizer = tokenizer

    return _shared_model, _shared_tokenizer


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Subclasses must implement: build_prompt() and parse_output().
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model, self.tokenizer = load_model(cfg)
        self.device = next(self.model.parameters()).device

    @abstractmethod
    def build_prompt(self, **kwargs) -> str:
        """Constructs the prompt string to send to the model."""
        pass

    @abstractmethod
    def parse_output(self, raw_output: str) -> dict:
        """Parses the raw model text output into a structured dict."""
        pass

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
        """
        Runs inference on the model with the given prompt.
        Returns only the newly generated tokens (not the prompt itself).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
        """Alias for generate() — preferred internal method name."""
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    def run(self, **kwargs) -> dict:
        """Default entry point (used only if subclass does not override run())."""
        prompt = self.build_prompt(**kwargs)
        raw_output = self._generate(prompt)
        return self.parse_output(raw_output, **kwargs)
