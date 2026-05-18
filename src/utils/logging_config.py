"""
Centralised logging setup for the PDG pipeline.

Usage
-----
    from src.utils.logging_config import get_logger, setup_pipeline_logging

    # In a module:
    logger = get_logger(__name__)

    # In train_pipeline.py (once, at startup):
    setup_pipeline_logging(log_file="logs/pipeline.log")
"""
import logging
import sys
from pathlib import Path


_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "pdg") -> logging.Logger:
    """Return (or create) a named logger with a stdout handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def setup_pipeline_logging(
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure the root 'pdg' logger.

    Parameters
    ----------
    log_file : str or None
        If provided, also write log output to this path (appended).
    level : int
        Logging level (default INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger("pdg")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
