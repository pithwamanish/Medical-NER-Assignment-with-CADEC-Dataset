"""
Text file loader for CADEC dataset.

This module provides utilities to load and process text files
from the CADEC dataset.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)


def load_text_file(text_file_path: Path) -> str:
    """
    Load text content from a single file.
    
    Args:
        text_file_path: Path to the .txt file
        
    Returns:
        Text content as string
        
    Raises:
        FileNotFoundError: If file does not exist
        UnicodeDecodeError: If file encoding is invalid
        
    Example:
        >>> from pathlib import Path
        >>> text_file = Path("cadec/text/sample.txt")
        >>> text = load_text_file(text_file)
        >>> print(f"Loaded {len(text)} characters")
    """
    if not text_file_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_file_path}")
    
    try:
        with open(text_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Normalize whitespace if configured
        if config.NORMALIZE_WHITESPACE:
            content = " ".join(content.split())
        
        logger.debug(f"Loaded text file: {text_file_path.name} ({len(content)} chars)")
        return content
    
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading {text_file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading text file {text_file_path}: {e}")
        raise


def load_all_texts(text_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Load all text files from the CADEC text directory.
    
    Args:
        text_dir: Directory containing text files. If None, uses config.TEXT_DIR
        
    Returns:
        Dictionary mapping filename (without extension) to text content
        
    Example:
        >>> texts = load_all_texts()
        >>> print(f"Loaded {len(texts)} text files")
        >>> print(f"Sample file: {list(texts.keys())[0]}")
    """
    if text_dir is None:
        text_dir = config.TEXT_DIR
    
    if not text_dir.exists():
        raise FileNotFoundError(f"Text directory not found: {text_dir}")
    
    texts: Dict[str, str] = {}
    text_files = list(text_dir.glob("*.txt"))
    
    logger.info(f"Loading {len(text_files)} text files from {text_dir}")
    
    for text_file in text_files:
        try:
            # Use filename without extension as key
            key = text_file.stem
            texts[key] = load_text_file(text_file)
        except Exception as e:
            logger.warning(f"Failed to load {text_file}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(texts)} text files")
    return texts


def get_text_for_entity(
    text: str, 
    start: int, 
    end: int
) -> Optional[str]:
    """
    Extract entity text from document text using character positions.
    
    Args:
        text: Full document text
        start: Start character position (inclusive)
        end: End character position (exclusive)
        
    Returns:
        Extracted entity text, or None if positions are invalid
        
    Example:
        >>> text = "I felt a bit drowsy after taking the medication."
        >>> entity_text = get_text_for_entity(text, 10, 20)
        >>> print(entity_text)  # "bit drowsy"
    """
    try:
        if start < 0 or end > len(text) or start >= end:
            logger.warning(f"Invalid positions: start={start}, end={end}, text_len={len(text)}")
            return None
        return text[start:end]
    except Exception as e:
        logger.error(f"Error extracting entity text: {e}")
        return None

