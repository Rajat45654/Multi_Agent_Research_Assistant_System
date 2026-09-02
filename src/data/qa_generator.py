"""
Q&A Dataset Generator.

Uses a local Mistral-7B model to generate Q&A pairs from text chunks.
Saves to data/processed/qa_pairs.json
"""

import json
import random
import torch
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.config import Config, load_config

logger = get_logger(__name__)

class QAGenerator:
    """Generates synthetic Q&A pairs using a local LLM."""
    
    def __init__(self, config: Config, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.processed_dir = Path(config.paths.processed_dir)
        self.chunks_file = self.processed_dir / "chunks.jsonl"
        self.qa_file = self.processed_dir / "qa_pairs.json"
        
        if not self.dry_run:
            self._init_model()
            
    def _init_model(self):
        """Initialize local Mistral model in bfloat16 (no quantization)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logger.error("transformers not installed. Cannot initialize model.")
            raise

        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        logger.info(f"Loading model {self.config.qa_generation.model_name} in bfloat16...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.qa_generation.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.qa_generation.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        logger.info("Model loaded successfully.")


    def _generate_prompt(self, context: str, difficulty: str) -> str:
        """Creates a prompt for the model based on difficulty."""
        if difficulty == "easy":
            instruction = "Generate a simple factual question that can be directly answered by the text. Then provide the exact answer."
        elif difficulty == "medium":
            instruction = "Generate a question that requires summarizing or reasoning about concepts in the text. Then provide a detailed answer."
        else: # hard
            instruction = "Generate a complex, analytical question that requires synthesizing multiple pieces of information from the text. Then provide a comprehensive answer."
            
        prompt = f"""<s>[INST] You are a helpful AI research assistant.
Your task is to generate a Question and Answer pair based ON THE PROVIDED CONTEXT ONLY.
{instruction}

Format your output exactly like this:
Question: <your question here>
Answer: <your answer here>

Context:
{context}
[/INST]"""
        return prompt

    def _parse_output(self, output: str) -> tuple[str, str]:
        """Parses model output into Question and Answer."""
        question, answer = "", ""
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
            elif line.startswith("Answer:"):
                # Answer might span multiple lines
                answer = "\n".join(lines[i:]).replace("Answer:", "").strip()
                break
        return question, answer

    def generate_for_chunk(self, chunk: Dict[str, Any], difficulty: str) -> Dict[str, Any]:
        """Generate a single Q&A pair from a chunk."""
        if self.dry_run:
            return {
                "question": f"Dummy {difficulty} question for {chunk['chunk_id']}?",
                "context_docs": [chunk['text']],
                "ground_truth_answer": f"Dummy answer based on context from {chunk['arxiv_id']}.",
                "difficulty_level": difficulty,
                "source_arxiv_id": chunk['arxiv_id']
            }
            
        prompt = self._generate_prompt(chunk['text'], difficulty)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.qa_generation.max_new_tokens,
            temperature=self.config.qa_generation.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Only get the new tokens generated
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        question, answer = self._parse_output(generated_text)
        
        if not question or not answer:
             # Fallback if model didn't follow format exactly
             logger.warning(f"Failed to parse model output for chunk {chunk['chunk_id']}")
             return None
             
        return {
            "question": question,
            "context_docs": [chunk['text']],
            "ground_truth_answer": answer,
            "difficulty_level": difficulty,
            "source_arxiv_id": chunk['arxiv_id']
        }

    def determine_difficulty(self) -> str:
        """Probabilistically select difficulty based on config."""
        r = random.random()
        splits = self.config.qa_generation.difficulty_split
        if r < splits.get("easy", 0.4):
            return "easy"
        elif r < splits.get("easy", 0.4) + splits.get("medium", 0.4):
            return "medium"
        return "hard"

    def _load_checkpoint(self) -> list:
        """Load existing QA pairs from checkpoint file if it exists."""
        checkpoint_file = self.processed_dir / "qa_pairs_checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            logger.info(f"Resuming from checkpoint: {len(existing)} pairs already generated.")
            return existing
        return []

    def _save_checkpoint(self, qa_pairs: list) -> None:
        """Save current progress to checkpoint file."""
        checkpoint_file = self.processed_dir / "qa_pairs_checkpoint.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f)

    def _get_processed_chunk_ids(self, qa_pairs: list) -> set:
        """Extract chunk IDs already processed so we skip them on resume."""
        # chunk_id is stored in source_arxiv_id + we use context_docs as proxy
        # We track via a separate set written alongside checkpoint
        ids_file = self.processed_dir / "qa_processed_ids.json"
        if ids_file.exists():
            with open(ids_file, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        return set()

    def _save_processed_ids(self, processed_ids: set) -> None:
        ids_file = self.processed_dir / "qa_processed_ids.json"
        with open(ids_file, 'w', encoding='utf-8') as f:
            json.dump(list(processed_ids), f)

    def process_all(self) -> None:
        """
        Process chunks and generate target number of Q&A pairs.

        Saves a checkpoint every 100 pairs so progress is never lost
        if the process is interrupted. On re-run, automatically resumes
        from the checkpoint.
        """
        if not self.chunks_file.exists():
            logger.error(f"Chunks file not found at {self.chunks_file}")
            return

        # Group chunks by arxiv_id to ensure distribution
        chunks_by_doc = {}
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                doc_id = chunk['arxiv_id']
                if doc_id not in chunks_by_doc:
                    chunks_by_doc[doc_id] = []
                chunks_by_doc[doc_id].append(chunk)

        doc_ids = list(chunks_by_doc.keys())
        if not doc_ids:
            logger.error("No chunks found to process.")
            return

        logger.info(f"Loaded chunks for {len(doc_ids)} documents.")

        # ── Resume from checkpoint if available ──────────────────────
        qa_pairs = self._load_checkpoint()
        processed_ids = self._get_processed_chunk_ids(qa_pairs)

        target = self.config.qa_generation.target_total
        pairs_per_doc = self.config.qa_generation.pairs_per_paper
        checkpoint_interval = 100  # save every N new pairs

        if len(qa_pairs) >= target:
            logger.info(f"Already have {len(qa_pairs)} pairs (target={target}). Nothing to do.")
            # Finalise output file from checkpoint
            with open(self.qa_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2)
            return

        pbar = tqdm(total=target, initial=len(qa_pairs), desc="Generating Q&A pairs")
        pairs_since_last_save = 0

        for doc_id in doc_ids:
            if len(qa_pairs) >= target:
                break

            doc_chunks = chunks_by_doc[doc_id]
            num_to_select = min(pairs_per_doc, len(doc_chunks))
            selected_chunks = random.sample(doc_chunks, num_to_select)

            for chunk in selected_chunks:
                if len(qa_pairs) >= target:
                    break

                chunk_id = chunk.get('chunk_id', '')

                # ── Skip already-processed chunks on resume ──────────
                if chunk_id in processed_ids:
                    continue

                difficulty = self.determine_difficulty()
                qa_pair = self.generate_for_chunk(chunk, difficulty)

                processed_ids.add(chunk_id)

                if qa_pair:
                    qa_pairs.append(qa_pair)
                    pbar.update(1)
                    pairs_since_last_save += 1

                    # ── Checkpoint every N pairs ──────────────────────
                    if pairs_since_last_save >= checkpoint_interval:
                        self._save_checkpoint(qa_pairs)
                        self._save_processed_ids(processed_ids)
                        logger.info(f"Checkpoint saved ({len(qa_pairs)}/{target} pairs).")
                        pairs_since_last_save = 0

        pbar.close()

        # ── Final save ────────────────────────────────────────────────
        with open(self.qa_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=2)

        # Clean up checkpoint files now that we're done
        checkpoint_file = self.processed_dir / "qa_pairs_checkpoint.json"
        ids_file = self.processed_dir / "qa_processed_ids.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        if ids_file.exists():
            ids_file.unlink()

        logger.info(f"Q&A generation complete. Saved {len(qa_pairs)} pairs to {self.qa_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from text chunks.")
    parser.add_argument("--dry_run", action="store_true", help="Run without loading the LLM (generates dummy data).")
    args = parser.parse_args()

    cfg = load_config()
    generator = QAGenerator(cfg, dry_run=args.dry_run)
    generator.process_all()
