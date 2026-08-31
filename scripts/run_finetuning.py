#!/usr/bin/env python3
"""
Script to execute LoRA fine-tuning of Mistral-7B.

Run from the research-assistant/ directory:
    python scripts/run_finetuning.py
    python scripts/run_finetuning.py --epochs 1  # quick test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.models.train import run_training

logger = get_logger("scripts.run_finetuning")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral-7B with LoRA.")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs.", default=None)
    parser.add_argument("--lr", type=float, help="Override learning rate.", default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging.")
    args = parser.parse_args()

    cfg = load_config()

    if args.epochs:
        cfg.finetuning.num_epochs = args.epochs
    if args.lr:
        cfg.finetuning.learning_rate = args.lr
    if args.no_wandb:
        cfg.finetuning.use_wandb = False

    logger.info("=== Starting Phase 2 Fine-Tuning ===")
    run_training(cfg)
    logger.info("=== Fine-Tuning Complete ===")


if __name__ == "__main__":
    main()
