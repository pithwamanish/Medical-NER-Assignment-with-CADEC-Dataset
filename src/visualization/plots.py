"""
Plotting utilities for NER evaluation and analysis.

This module provides functions for creating various visualizations
related to NER performance, entity distributions, and comparisons.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
sns.set_palette(config.COLOR_PALETTE)


def plot_metrics(
    metrics: Dict[str, Dict[str, float]],
    label_types: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot precision, recall, and F1 scores for each label.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics
        label_types: List of label types to plot
        save_path: Path to save figure. If None and config.SAVE_FIGURES is True,
                   saves to results directory
        show: Whether to display the plot
        
    Example:
        >>> plot_metrics(metrics, save_path=Path("metrics.png"))
    """
    label_types = label_types or config.LABEL_TYPES
    
    # Prepare data
    labels = []
    precision = []
    recall = []
    f1 = []
    
    for label in label_types:
        if label in metrics:
            labels.append(label)
            precision.append(metrics[label].get("precision", 0))
            recall.append(metrics[label].get("recall", 0))
            f1.append(metrics[label].get("f1", 0))
    
    if not labels:
        logger.warning("No metrics to plot")
        return
    
    # Create plot
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.DPI)
    ax.bar(x - width, precision, width, label="Precision", alpha=0.8)
    ax.bar(x, recall, width, label="Recall", alpha=0.8)
    ax.bar(x + width, f1, width, label="F1", alpha=0.8)
    
    ax.set_xlabel("Label Type")
    ax.set_ylabel("Score")
    ax.set_title("NER Performance Metrics by Label")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None and config.SAVE_FIGURES:
        save_path = config.RESULTS_DIR / "metrics_plot.png"
    
    if save_path:
        fig.savefig(save_path, dpi=config.DPI, bbox_inches="tight")
        logger.info(f"Saved metrics plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_confusion_matrix(
    metrics: Dict[str, Dict[str, float]],
    label_types: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot confusion matrix heatmap.
    
    Args:
        metrics: Dictionary of metrics
        label_types: List of label types
        save_path: Path to save figure
        show: Whether to display the plot
    """
    label_types = label_types or config.LABEL_TYPES
    
    # Prepare confusion matrix data
    data = []
    for label in label_types:
        if label in metrics:
            m = metrics[label]
            data.append({
                "Label": label,
                "TP": m.get("tp", 0),
                "FP": m.get("fp", 0),
                "FN": m.get("fn", 0),
            })
    
    if not data:
        logger.warning("No data for confusion matrix")
        return
    
    df = pd.DataFrame(data).set_index("Label")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.DPI)
    sns.heatmap(
        df,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_title("Confusion Matrix (TP, FP, FN by Label)")
    plt.tight_layout()
    
    if save_path is None and config.SAVE_FIGURES:
        save_path = config.RESULTS_DIR / "confusion_matrix.png"
    
    if save_path:
        fig.savefig(save_path, dpi=config.DPI, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_entity_distribution(
    entities: List[Dict],
    label_types: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot distribution of entity labels.
    
    Args:
        entities: List of entity dictionaries
        label_types: List of label types to include
        save_path: Path to save figure
        show: Whether to display the plot
    """
    label_types = label_types or config.LABEL_TYPES
    
    # Count entities by label
    label_counts = {}
    for entity in entities:
        label = entity.get("label", "O")
        if label in label_types:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    if not label_counts:
        logger.warning("No entities to plot")
        return
    
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.DPI)
    ax.bar(labels, counts, alpha=0.8)
    ax.set_xlabel("Label Type")
    ax.set_ylabel("Count")
    ax.set_title("Entity Distribution by Label")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path is None and config.SAVE_FIGURES:
        save_path = config.RESULTS_DIR / "entity_distribution.png"
    
    if save_path:
        fig.savefig(save_path, dpi=config.DPI, bbox_inches="tight")
        logger.info(f"Saved entity distribution plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_performance_comparison(
    metrics_list: List[Dict[str, Dict[str, float]]],
    labels: List[str],
    metric: str = "f1",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Compare performance across multiple runs/models.
    
    Args:
        metrics_list: List of metrics dictionaries (one per model/run)
        labels: List of labels for each metrics dict
        metric: Metric to compare ('precision', 'recall', 'f1')
        save_path: Path to save figure
        show: Whether to display the plot
    """
    if len(metrics_list) != len(labels):
        raise ValueError("metrics_list and labels must have same length")
    
    label_types = config.LABEL_TYPES
    
    # Prepare data
    comparison_data = []
    for label_type in label_types:
        row = {"Label": label_type}
        for i, metrics in enumerate(metrics_list):
            if label_type in metrics:
                row[labels[i]] = metrics[label_type].get(metric, 0)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data).set_index("Label")
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.DPI)
    df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{metric.capitalize()} Comparison Across Models/Runs")
    ax.legend(title="Model/Run")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_path is None and config.SAVE_FIGURES:
        save_path = config.RESULTS_DIR / f"{metric}_comparison.png"
    
    if save_path:
        fig.savefig(save_path, dpi=config.DPI, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)

