"""
Base Agent class that all agents inherit from.

Provides a standard interface for running inference across dual backends:
1. 'local'  : Fine-tuned Mistral-7B loaded locally on NVIDIA GPU via PyTorch
2. 'gemini' : Google Gemini Pro / Flash Cloud API (zero GPU required, runs on CPU/laptop)
"""

import os
import torch
from pathlib import Path
from abc import ABC, abstractmethod

from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)

# Shared local model instance (loaded once, reused by all agents)
_shared_model = None
_shared_tokenizer = None

# Shared Gemini client instance (initialized once, reused by all agents)
_shared_gemini_client = None


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

    logger.info("Loading local Mistral-7B model for agent inference...")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

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
    if hasattr(model, "config"):
        model.config.use_cache = True
    _shared_model = model
    _shared_tokenizer = tokenizer

    return _shared_model, _shared_tokenizer


class GeminiRestAdapter:
    """Fallback REST adapter for direct Gemini API calls via HTTP."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self.url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )

    def generate_content(self, prompt: str, generation_config=None):
        import requests
        temp = 0.3
        max_tokens = 1024
        if generation_config is not None:
            temp = getattr(generation_config, "temperature", 0.3)
            max_tokens = getattr(generation_config, "max_output_tokens", 1024)

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temp,
                "maxOutputTokens": max_tokens,
            }
        }
        res = requests.post(self.url, json=payload, timeout=90)
        res.raise_for_status()
        data = res.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return type("Resp", (), {"text": ""})()
        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return type("Resp", (), {"text": text})()


def init_gemini(cfg: Config):
    """
    Initializes and caches the Google Gemini client.
    Reads API key from config or environment variable.
    """
    global _shared_gemini_client
    if _shared_gemini_client is not None:
        return _shared_gemini_client

    api_key = (cfg.llm.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")).strip()

    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError(
            "\n" + "=" * 65 + "\n"
            "  ❌ GEMINI_API_KEY IS MISSING OR NOT CONFIGURED!\n"
            "=" * 65 + "\n"
            "  The research assistant is set to backend: 'gemini', but no\n"
            "  valid Gemini API key was found.\n\n"
            "  How to fix this:\n"
            "  1. Get a free API key at: https://aistudio.google.com/app/apikey\n"
            "  2. Paste it in your .env file:\n"
            "         GEMINI_API_KEY=AIzaSy...\n"
            "     OR export it in your terminal:\n"
            "         export GEMINI_API_KEY=\"AIzaSy...\"\n"
            "=" * 65 + "\n"
        )

    model_name = (cfg.llm.gemini_model or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")).strip()
    logger.info(f"Initializing Gemini client with model '{model_name}'...")

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _shared_gemini_client = genai.GenerativeModel(model_name)
        logger.info(f"Google Gemini SDK initialized successfully (model: {model_name}).")
    except Exception as e:
        logger.warning(f"google.generativeai SDK init note: {e}. Using GeminiRestAdapter.")
        _shared_gemini_client = GeminiRestAdapter(api_key=api_key, model=model_name)

    return _shared_gemini_client


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Subclasses must implement: build_prompt() and parse_output().

    Supports dual backends:
      - 'local' : Fine-tuned Mistral-7B on NVIDIA GPU via PyTorch
      - 'gemini': Google Gemini Pro / Flash API (zero GPU required)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backend = getattr(cfg, "llm", None) and getattr(cfg.llm, "backend", "local") or "local"
        self.backend = self.backend.lower().strip()

        if self.backend == "local":
            self.model, self.tokenizer = load_model(cfg)
            self.device = next(self.model.parameters()).device
            self.gemini_model = None
        elif self.backend == "gemini":
            self.gemini_model = init_gemini(cfg)
            self.model, self.tokenizer = None, None
            self.device = "cpu"
        else:
            raise ValueError(f"Unknown LLM backend: '{self.backend}'. Must be 'local' or 'gemini'.")

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
        Runs inference on either local model (GPU) or Gemini API (Cloud).
        Returns only the newly generated text.
        """
        if self.backend == "gemini":
            return self._generate_gemini(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        else:
            return self._generate_local(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    def _generate_local(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
        """Runs local PyTorch inference on GPU."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                repetition_penalty=1.15,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _generate_gemini(self, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.3) -> str:
        """Generates text via Google Gemini API."""
        # Clean Mistral instruction tags so Gemini receives clean natural instructions
        clean_prompt = (
            prompt.replace("<s>", "")
            .replace("</s>", "")
            .replace("[INST]", "")
            .replace("[/INST]", "")
            .strip()
        )
        try:
            import google.generativeai as genai
            config = genai.types.GenerationConfig(
                max_output_tokens=max_new_tokens,
                temperature=temperature,
            )
            response = self.gemini_model.generate_content(clean_prompt, generation_config=config)
            return response.text.strip()
        except Exception as e:
            # Fallback to direct call if SDK has an issue
            if hasattr(self.gemini_model, "generate_content"):
                try:
                    response = self.gemini_model.generate_content(clean_prompt)
                    return getattr(response, "text", "").strip()
                except Exception:
                    pass
            logger.error(f"Gemini generation error: {e}", exc_info=True)
            raise RuntimeError(f"Gemini API generation failed: {e}") from e

    def _generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
        """Alias for generate() — preferred internal method name."""
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    def run(self, **kwargs) -> dict:
        """Default entry point (used only if subclass does not override run())."""
        prompt = self.build_prompt(**kwargs)
        raw_output = self._generate(prompt)
        return self.parse_output(raw_output, **kwargs)
