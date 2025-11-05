"""
Logging utilities for CADEC NER Project.

This module provides centralized logging configuration and setup.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import config


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    log_to_file: Optional[bool] = None,
    log_to_console: Optional[bool] = None,
) -> None:
    """
    Configure logging for the project.
    
    Sets up logging with both file and console handlers, using
    the format and level specified in config.
    
    Args:
        log_level: Logging level. If None, uses config.LOG_LEVEL
        log_file: Path to log file. If None, uses config.LOG_FILE
        log_to_file: Whether to log to file. If None, uses config.LOG_TO_FILE
        log_to_console: Whether to log to console. If None, uses config.LOG_TO_CONSOLE
        
    Example:
        >>> setup_logging()
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Logging configured")
    """
    log_level = log_level or config.LOG_LEVEL
    log_file = log_file or config.LOG_FILE
    log_to_file = log_to_file if log_to_file is not None else config.LOG_TO_FILE
    log_to_console = log_to_console if log_to_console is not None else config.LOG_TO_CONSOLE
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logging.info("Logging configured successfully")
    logging.info(f"Log level: {log_level}")
    if log_to_file:
        logging.info(f"Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Using logger")
    """
    return logging.getLogger(name)

