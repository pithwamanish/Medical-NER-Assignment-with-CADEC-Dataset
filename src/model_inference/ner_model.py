"""
NER model wrapper for token classification.

This module provides a unified interface for loading and using
Named Entity Recognition models, with support for BioBERT and
other transformer-based models.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

import config

logger = logging.getLogger(__name__)


class NERModel:
    """
    Wrapper for NER token classification models.
    
    This class provides a unified interface for loading and running
    inference with transformer-based NER models. Supports caching,
    batch processing, and GPU acceleration.
    
    Attributes:
        model_name: Name/path of the model
        tokenizer: Tokenizer instance
        model: Model instance
        device: Computation device (cuda or cpu)
        pipeline: Hugging Face pipeline for inference
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize NER model.
        
        Args:
            model_name: Hugging Face model identifier. If None, uses
                       config.LLM_MODEL_NAME
            device: Computation device ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name or config.LLM_MODEL_NAME
        self.device = device or config.get_device()
        self.cache_dir = cache_dir or config.MODELS_DIR
        
        logger.info(f"Initializing NER model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
            )
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if "cuda" in self.device else -1,
                aggregation_strategy="simple",  # Aggregate subword tokens
            )
            
            logger.info("NER model initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize NER model: {e}")
            raise
    
    def predict(self, text: str) -> List[Dict]:
        """
        Predict entities in a single text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of entity dictionaries with:
            - 'entity_group': Entity label (e.g., 'ADR', 'Drug')
            - 'score': Confidence score
            - 'word': Entity text
            - 'start': Start character position
            - 'end': End character position
            
        Example:
            >>> model = NERModel()
            >>> text = "I felt drowsy after taking aspirin."
            >>> entities = model.predict(text)
            >>> print(entities)
        """
        if not text.strip():
            logger.warning("Empty text provided for prediction")
            return []
        
        try:
            results = self.pipeline(text)
            logger.debug(f"Predicted {len(results)} entities in text")
            return results
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return []
    
    def predict_batch(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None
    ) -> List[List[Dict]]:
        """
        Predict entities for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing. If None, uses
                       config.BATCH_SIZE
            
        Returns:
            List of prediction results, one per input text
            
        Example:
            >>> model = NERModel()
            >>> texts = ["I felt drowsy.", "Aspirin helped."]
            >>> results = model.predict_batch(texts)
            >>> print(f"Processed {len(results)} texts")
        """
        if not texts:
            return []
        
        batch_size = batch_size or config.BATCH_SIZE
        results = []
        
        logger.info(f"Processing {len(texts)} texts in batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                batch_results = self.pipeline(batch)
                # Pipeline returns nested list for batch input
                if isinstance(batch_results[0], list):
                    results.extend(batch_results)
                else:
                    results.append(batch_results)
            except Exception as e:
                logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
                # Add empty results for failed batch
                results.extend([[]] * len(batch))
        
        logger.info(f"Completed batch processing: {len(results)} results")
        return results
    
    def __del__(self):
        """Cleanup resources."""
        # Clear cache if needed
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_ner_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> NERModel:
    """
    Load and return an NER model instance.
    
    Convenience function for creating NERModel instances with default
    configuration.
    
    Args:
        model_name: Model identifier. If None, uses config.LLM_MODEL_NAME
        device: Computation device. If None, auto-detects from config
        
    Returns:
        Initialized NERModel instance
        
    Example:
        >>> model = load_ner_model()
        >>> entities = model.predict("I felt nauseous.")
    """
    return NERModel(model_name=model_name, device=device)

