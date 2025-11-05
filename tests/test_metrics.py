"""
Unit tests for evaluation metrics.

Tests precision, recall, F1 calculation and span matching functions.
"""

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    span_match,
    precision_recall_f1,
    confusion_matrix,
    calculate_metrics,
)


class TestSpanMatching:
    """Test suite for span matching functions."""
    
    def test_exact_match(self):
        """Test exact span matching."""
        assert span_match((10, 20), (10, 20), strategy="exact") is True
        assert span_match((10, 20), (10, 21), strategy="exact") is False
        assert span_match((10, 20), (11, 20), strategy="exact") is False
    
    def test_overlap_match(self):
        """Test overlap-based span matching."""
        assert span_match((10, 20), (15, 25), strategy="overlap") is True  # Overlap
        assert span_match((10, 20), (20, 30), strategy="overlap") is False  # Adjacent, no overlap
        assert span_match((10, 20), (25, 35), strategy="overlap") is False  # No overlap
        assert span_match((10, 20), (5, 15), strategy="overlap") is True  # Overlap
    
    def test_invalid_positions(self):
        """Test handling of invalid span positions."""
        # Invalid: end before start
        assert span_match((20, 10), (10, 20), strategy="exact") is False


class TestPrecisionRecallF1:
    """Test suite for precision, recall, F1 calculation."""
    
    def test_perfect_scores(self):
        """Test perfect prediction (no FP or FN)."""
        p, r, f1 = precision_recall_f1(tp=100, fp=0, fn=0)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0
    
    def test_zero_precision(self):
        """Test case with zero precision (all predictions wrong)."""
        p, r, f1 = precision_recall_f1(tp=0, fp=100, fn=0)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0
    
    def test_zero_recall(self):
        """Test case with zero recall (no predictions)."""
        p, r, f1 = precision_recall_f1(tp=0, fp=0, fn=100)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0
    
    def test_balanced_scores(self):
        """Test balanced precision and recall."""
        p, r, f1 = precision_recall_f1(tp=80, fp=20, fn=20)
        assert abs(p - 0.8) < 0.01
        assert abs(r - 0.8) < 0.01
        assert abs(f1 - 0.8) < 0.01
    
    def test_division_by_zero(self):
        """Test handling of division by zero cases."""
        # No predictions at all
        p, r, f1 = precision_recall_f1(tp=0, fp=0, fn=0)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0


class TestConfusionMatrix:
    """Test suite for confusion matrix calculation."""
    
    def test_perfect_match(self):
        """Test confusion matrix with perfect predictions."""
        pred_entities = [
            {"label": "ADR", "start": 10, "end": 20},
            {"label": "Drug", "start": 25, "end": 30},
        ]
        gold_entities = [
            {"label": "ADR", "start": 10, "end": 20},
            {"label": "Drug", "start": 25, "end": 30},
        ]
        
        cm = confusion_matrix(pred_entities, gold_entities)
        
        assert cm["ADR"]["tp"] == 1
        assert cm["ADR"]["fp"] == 0
        assert cm["ADR"]["fn"] == 0
        assert cm["Drug"]["tp"] == 1
        assert cm["Drug"]["fp"] == 0
        assert cm["Drug"]["fn"] == 0
    
    def test_false_positives(self):
        """Test confusion matrix with false positives."""
        pred_entities = [
            {"label": "ADR", "start": 10, "end": 20},
            {"label": "ADR", "start": 50, "end": 60},  # FP
        ]
        gold_entities = [
            {"label": "ADR", "start": 10, "end": 20},
        ]
        
        cm = confusion_matrix(pred_entities, gold_entities)
        
        assert cm["ADR"]["tp"] == 1
        assert cm["ADR"]["fp"] == 1
        assert cm["ADR"]["fn"] == 0
    
    def test_false_negatives(self):
        """Test confusion matrix with false negatives."""
        pred_entities = [
            {"label": "ADR", "start": 10, "end": 20},
        ]
        gold_entities = [
            {"label": "ADR", "start": 10, "end": 20},
            {"label": "ADR", "start": 50, "end": 60},  # FN
        ]
        
        cm = confusion_matrix(pred_entities, gold_entities)
        
        assert cm["ADR"]["tp"] == 1
        assert cm["ADR"]["fp"] == 0
        assert cm["ADR"]["fn"] == 1


class TestCalculateMetrics:
    """Test suite for comprehensive metrics calculation."""
    
    def test_overall_metrics(self):
        """Test calculation of overall metrics."""
        pred_entities = [
            {"label": "ADR", "start": 10, "end": 20},
            {"label": "Drug", "start": 25, "end": 30},
        ]
        gold_entities = [
            {"label": "ADR", "start": 10, "end": 20},
            {"label": "Drug", "start": 25, "end": 30},
        ]
        
        metrics = calculate_metrics(pred_entities, gold_entities)
        
        assert "overall" in metrics
        assert metrics["overall"]["tp"] == 2
        assert metrics["overall"]["fp"] == 0
        assert metrics["overall"]["fn"] == 0
        assert metrics["overall"]["f1"] == 1.0
    
    def test_per_label_metrics(self):
        """Test calculation of per-label metrics."""
        pred_entities = [
            {"label": "ADR", "start": 10, "end": 20},
        ]
        gold_entities = [
            {"label": "ADR", "start": 10, "end": 20},
            {"label": "Drug", "start": 25, "end": 30},  # FN for Drug
        ]
        
        metrics = calculate_metrics(pred_entities, gold_entities)
        
        assert "ADR" in metrics
        assert metrics["ADR"]["tp"] == 1
        assert metrics["ADR"]["fn"] == 0
        assert "Drug" in metrics
        assert metrics["Drug"]["tp"] == 0
        assert metrics["Drug"]["fn"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

