"""
Utility modules for CADEC NER Project.

This module provides shared utilities including logging setup,
caching, and other helper functions.
"""

from .logging_utils import setup_logging, get_logger

__all__ = [
    "setup_logging",
    "get_logger",
]

