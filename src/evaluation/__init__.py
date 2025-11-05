"""
Evaluation module for CADEC NER Project.

This module provides functionality for:
- Calculating precision, recall, F1 scores
- Span-based matching strategies
- Confusion matrices and classification reports
- Performance metrics aggregation
"""

from .metrics import (
    calculate_metrics,
    span_match,
    precision_recall_f1,
    confusion_matrix,
    ClassificationReport,
)

__all__ = [
    "calculate_metrics",
    "span_match",
    "precision_recall_f1",
    "confusion_matrix",
    "ClassificationReport",
]

