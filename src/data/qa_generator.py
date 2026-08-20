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
        """Initialize local Mistral model in 4-bit."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            logger.error("transformers or bitsandbytes not installed. Cannot initialize model.")
            raise
            
        logger.info(f"Loading model {self.config.qa_generation.model_name} in 4-bit mode...")
        
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

    def process_all(self) -> None:
        """Process chunks and generate target number of Q&A pairs."""
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
        
        qa_pairs = []
        target = self.config.qa_generation.target_total
        pairs_per_doc = self.config.qa_generation.pairs_per_paper
        
        pbar = tqdm(total=target, desc="Generating Q&A pairs")
        
        # Simple loop: take a few chunks from each doc
        # In a real scenario, we might want to prioritize specific types of chunks
        for doc_id in doc_ids:
            if len(qa_pairs) >= target:
                break
                
            doc_chunks = chunks_by_doc[doc_id]
            # Select random chunks from this document
            num_to_select = min(pairs_per_doc, len(doc_chunks))
            selected_chunks = random.sample(doc_chunks, num_to_select)
            
            for chunk in selected_chunks:
                if len(qa_pairs) >= target:
                    break
                    
                difficulty = self.determine_difficulty()
                qa_pair = self.generate_for_chunk(chunk, difficulty)
                
                if qa_pair:
                    qa_pairs.append(qa_pair)
                    pbar.update(1)
                    
        pbar.close()
        
        with open(self.qa_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=2)
            
        logger.info(f"Q&A generation complete. Saved {len(qa_pairs)} pairs to {self.qa_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from text chunks.")
    parser.add_argument("--dry_run", action="store_true", help="Run without loading the LLM (generates dummy data).")
    args = parser.parse_args()
    
    cfg = load_config()
    generator = QAGenerator(cfg, dry_run=args.dry_run)
    generator.process_all()
