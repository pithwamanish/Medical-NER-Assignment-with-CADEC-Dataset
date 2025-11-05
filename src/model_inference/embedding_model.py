"""
Embedding model for semantic similarity matching.

This module provides functionality for generating embeddings using
sentence transformers, useful for entity normalization and similarity
search.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for sentence transformer embedding models.
    
    This class provides functionality for generating embeddings
    from text, useful for semantic similarity matching in entity
    normalization tasks.
    
    Attributes:
        model_name: Name/path of the embedding model
        model: SentenceTransformer instance
        device: Computation device
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: Hugging Face model identifier. If None, uses
                       config.EMBEDDING_MODEL_NAME
            device: Computation device ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.device = device or config.get_device()
        self.cache_dir = cache_dir or config.MODELS_DIR
        
        logger.info(f"Initializing embedding model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        try:
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir),
                device=self.device,
            )
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing. If None, uses
                       config.EMBEDDING_BATCH_SIZE
            normalize: Whether to normalize embeddings to unit length
            
        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
            
        Example:
            >>> model = EmbeddingModel()
            >>> embeddings = model.encode(["drowsy", "sleepy"])
            >>> print(f"Embedding shape: {embeddings.shape}")
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 10,
            )
            logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def similarity(
        self,
        query_embeddings: np.ndarray,
        candidate_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate cosine similarity between query and candidate embeddings.
        
        Args:
            query_embeddings: Query embeddings (n_queries, dim)
            candidate_embeddings: Candidate embeddings (n_candidates, dim)
            
        Returns:
            Similarity matrix (n_queries, n_candidates)
            
        Example:
            >>> model = EmbeddingModel()
            >>> query_emb = model.encode(["drowsy"])
            >>> candidates_emb = model.encode(["sleepy", "alert"])
            >>> similarities = model.similarity(query_emb, candidates_emb)
        """
        # Cosine similarity: dot product of normalized vectors
        return np.dot(query_embeddings, candidate_embeddings.T)


def load_embedding_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> EmbeddingModel:
    """
    Load and return an embedding model instance.
    
    Convenience function for creating EmbeddingModel instances with
    default configuration.
    
    Args:
        model_name: Model identifier. If None, uses config.EMBEDDING_MODEL_NAME
        device: Computation device. If None, auto-detects from config
        
    Returns:
        Initialized EmbeddingModel instance
        
    Example:
        >>> model = load_embedding_model()
        >>> embeddings = model.encode("medical term")
    """
    return EmbeddingModel(model_name=model_name, device=device)

