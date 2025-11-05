"""
Evaluation metrics for NER tasks.

This module provides functions for calculating precision, recall, F1,
and other evaluation metrics for named entity recognition tasks.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
from seqeval.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import config

logger = logging.getLogger(__name__)


def span_match(
    pred_span: Tuple[int, int],
    gold_span: Tuple[int, int],
    strategy: Optional[str] = None,
) -> bool:
    """
    Check if prediction and gold standard spans match.
    
    Supports multiple matching strategies:
    - "exact": Exact match required (strict)
    - "overlap": Any overlap considered match (lenient)
    - "relaxed": Overlap threshold-based matching
    
    Args:
        pred_span: Prediction span (start, end)
        gold_span: Gold standard span (start, end)
        strategy: Matching strategy. If None, uses config.SPAN_MATCH_STRATEGY
        
    Returns:
        True if spans match according to strategy
        
    Example:
        >>> span_match((10, 20), (10, 20), "exact")  # True
        >>> span_match((10, 20), (15, 25), "overlap")  # True
    """
    strategy = strategy or config.SPAN_MATCH_STRATEGY
    pred_start, pred_end = pred_span
    gold_start, gold_end = gold_span
    
    if strategy == "exact":
        return pred_start == gold_start and pred_end == gold_end
    
    elif strategy == "overlap":
        # Check for any overlap
        return not (pred_end <= gold_start or pred_start >= gold_end)
    
    elif strategy == "relaxed":
        # Calculate overlap ratio
        overlap_start = max(pred_start, gold_start)
        overlap_end = min(pred_end, gold_end)
        overlap_len = max(0, overlap_end - overlap_start)
        
        pred_len = pred_end - pred_start
        gold_len = gold_end - gold_start
        union_len = pred_len + gold_len - overlap_len
        
        if union_len == 0:
            return False
        
        overlap_ratio = overlap_len / union_len
        return overlap_ratio >= config.OVERLAP_THRESHOLD
    
    else:
        logger.warning(f"Unknown strategy: {strategy}, using exact match")
        return pred_start == gold_start and pred_end == gold_end


def precision_recall_f1(
    tp: int,
    fp: int,
    fn: int,
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        
    Returns:
        Tuple of (precision, recall, f1)
        
    Example:
        >>> p, r, f = precision_recall_f1(80, 20, 10)
        >>> print(f"F1: {f:.3f}")
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def confusion_matrix(
    pred_entities: List[Dict],
    gold_entities: List[Dict],
    label_types: Optional[List[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Calculate confusion matrix for entity predictions.
    
    Args:
        pred_entities: List of predicted entities with 'label', 'start', 'end'
        gold_entities: List of gold entities with 'label', 'start', 'end'
        label_types: List of label types to evaluate. If None, uses
                    config.LABEL_TYPES
        
    Returns:
        Dictionary mapping label to {'tp': int, 'fp': int, 'fn': int}
        
    Example:
        >>> preds = [{'label': 'ADR', 'start': 10, 'end': 20}]
        >>> golds = [{'label': 'ADR', 'start': 10, 'end': 20}]
        >>> cm = confusion_matrix(preds, golds)
        >>> print(cm['ADR']['tp'])  # 1
    """
    label_types = label_types or config.LABEL_TYPES
    
    # Convert entities to span sets per label
    pred_spans_by_label = defaultdict(set)
    gold_spans_by_label = defaultdict(set)
    
    for entity in pred_entities:
        label = entity.get("label", "O")
        if label in label_types:
            span = (entity["start"], entity["end"])
            pred_spans_by_label[label].add(span)
    
    for entity in gold_entities:
        label = entity.get("label", "O")
        if label in label_types:
            span = (entity["start"], entity["end"])
            gold_spans_by_label[label].add(span)
    
    # Calculate TP, FP, FN per label
    results = {}
    for label in label_types:
        pred_spans = pred_spans_by_label[label]
        gold_spans = gold_spans_by_label[label]
        
        tp = len(pred_spans & gold_spans)
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)
        
        results[label] = {"tp": tp, "fp": fp, "fn": fn}
    
    return results


def calculate_metrics(
    pred_entities: List[Dict],
    gold_entities: List[Dict],
    label_types: Optional[List[str]] = None,
    strategy: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        pred_entities: List of predicted entities
        gold_entities: List of gold standard entities
        label_types: Label types to evaluate
        strategy: Span matching strategy
        
    Returns:
        Dictionary mapping label to metrics dict with 'precision', 'recall',
        'f1', and optionally per-label breakdowns
        
    Example:
        >>> metrics = calculate_metrics(preds, golds)
        >>> print(f"Overall F1: {metrics['overall']['f1']:.3f}")
    """
    label_types = label_types or config.LABEL_TYPES
    
    # Calculate confusion matrix
    cm = confusion_matrix(pred_entities, gold_entities, label_types)
    
    # Aggregate totals
    total_tp = sum(cm[label]["tp"] for label in label_types)
    total_fp = sum(cm[label]["fp"] for label in label_types)
    total_fn = sum(cm[label]["fn"] for label in label_types)
    
    # Overall metrics
    overall_p, overall_r, overall_f1 = precision_recall_f1(
        total_tp, total_fp, total_fn
    )
    
    results = {
        "overall": {
            "precision": overall_p,
            "recall": overall_r,
            "f1": overall_f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        }
    }
    
    # Per-label metrics
    for label in label_types:
        tp = cm[label]["tp"]
        fp = cm[label]["fp"]
        fn = cm[label]["fn"]
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        
        results[label] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    
    return results


class ClassificationReport:
    """
    Classification report generator for NER evaluation.
    
    This class provides a structured way to generate and display
    classification reports with per-label and overall metrics.
    """
    
    def __init__(
        self,
        metrics: Dict[str, Dict[str, float]],
        label_types: Optional[List[str]] = None,
    ):
        """
        Initialize classification report.
        
        Args:
            metrics: Dictionary of metrics from calculate_metrics
            label_types: List of label types (for ordering)
        """
        self.metrics = metrics
        self.label_types = label_types or config.LABEL_TYPES
    
    def __str__(self) -> str:
        """Generate string representation of report."""
        lines = ["=" * 60, "Classification Report", "=" * 60, ""]
        
        # Overall metrics
        overall = self.metrics.get("overall", {})
        lines.append("Overall Metrics:")
        lines.append(f"  Precision: {overall.get('precision', 0):.4f}")
        lines.append(f"  Recall:    {overall.get('recall', 0):.4f}")
        lines.append(f"  F1 Score:  {overall.get('f1', 0):.4f}")
        lines.append(f"  TP: {overall.get('tp', 0)}, FP: {overall.get('fp', 0)}, "
                    f"FN: {overall.get('fn', 0)}")
        lines.append("")
        
        # Per-label metrics
        lines.append("Per-Label Metrics:")
        lines.append(f"{'Label':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} "
                    f"{'TP':<6} {'FP':<6} {'FN':<6}")
        lines.append("-" * 60)
        
        for label in self.label_types:
            if label in self.metrics:
                m = self.metrics[label]
                lines.append(
                    f"{label:<12} {m.get('precision', 0):<12.4f} "
                    f"{m.get('recall', 0):<12.4f} {m.get('f1', 0):<12.4f} "
                    f"{m.get('tp', 0):<6} {m.get('fp', 0):<6} {m.get('fn', 0):<6}"
                )
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return self.metrics

