"""
Data loading module for CADEC dataset.

This module provides functions to load and parse:
- Text files from CADEC dataset
- Annotation files (original, SCT, MedDRA)
- Entity extraction and normalization
"""

from .annotation_parser import (
    load_ground_truth,
    parse_snomed_annotations,
    parse_meddra_annotations,
    AnnotationParser,
)
from .text_loader import load_text_file, load_all_texts

__all__ = [
    "load_ground_truth",
    "parse_snomed_annotations",
    "parse_meddra_annotations",
    "AnnotationParser",
    "load_text_file",
    "load_all_texts",
]

