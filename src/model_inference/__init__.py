"""
Model inference module for CADEC NER Project.

This module provides functionality for:
- Loading and initializing NER models (BioBERT, Medical-NER, etc.)
- Running inference on text
- Batch processing for efficiency
- Model caching and optimization
"""

from .ner_model import NERModel, load_ner_model
from .embedding_model import EmbeddingModel, load_embedding_model

__all__ = [
    "NERModel",
    "load_ner_model",
    "EmbeddingModel",
    "load_embedding_model",
]

