"""
Visualization module for CADEC NER Project.

This module provides utilities for:
- Plotting evaluation metrics
- Visualizing entity distributions
- Generating performance charts
- Creating confusion matrices
"""

from .plots import (
    plot_metrics,
    plot_confusion_matrix,
    plot_entity_distribution,
    plot_performance_comparison,
)

__all__ = [
    "plot_metrics",
    "plot_confusion_matrix",
    "plot_entity_distribution",
    "plot_performance_comparison",
]

