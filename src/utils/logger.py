"""
Structured logger for the research assistant system.

Writes to both console (stdout) and a dated log file simultaneously.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a logger that writes to both console and file.

    Args:
        name: Logger name (use __name__ from calling module).
        log_dir: Directory to write log files into.
        level: Logging level (default: INFO).

    Returns:
        Configured :class:`logging.Logger` instance.

    Example::

        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Starting data collection...")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ─────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler ─────────────────────────────────────────────
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"research_assistant_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
