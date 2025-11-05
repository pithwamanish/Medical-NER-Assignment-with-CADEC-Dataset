"""
Configuration module for CADEC NER Project.

This module centralizes all hyperparameters, paths, model configurations,
and runtime settings for the medical Named Entity Recognition project.
"""

import os
from pathlib import Path
from typing import List, Optional

# ============================================================================
# Project Paths
# ============================================================================

# Base directory for the project (parent of this config file)
PROJECT_ROOT = Path(__file__).parent.absolute()

# CADEC dataset directories
BASE_DIR = PROJECT_ROOT / "cadec"
TEXT_DIR = BASE_DIR / "text"
ORIGINAL_DIR = BASE_DIR / "original"
SCT_DIR = BASE_DIR / "sct"
MEDDRA_DIR = BASE_DIR / "meddra"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
CACHE_DIR = OUTPUT_DIR / "cache"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create output directories if they don't exist
for dir_path in [OUTPUT_DIR, MODELS_DIR, RESULTS_DIR, CACHE_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Model Configuration
# ============================================================================

# LLM Model Selection
# Options:
#   - 'dmis-lab/biobert-base-cased-v1.1': BioBERT (proven for medical NER)
#   - 'blaze999/Medical-NER': Domain-specific token classification
#   - 'bert-base-cased': General purpose BERT (baseline)
LLM_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

# Embedding Model Selection
# Options:
#   - 'sentence-transformers/all-MiniLM-L6-v2': Fast and effective (recommended)
#   - 'sentence-transformers/paraphrase-MiniLM-L6-v2': Alternative fast option
#   - 'dmis-lab/biobert-base-cased-v1.1': Medical domain embeddings (slower)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Model Hyperparameters
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 1

# Token Classification Labels (BIO tagging scheme)
BIO_LABELS = [
    "O",  # Outside
    "B-ADR", "I-ADR",  # Adverse Drug Reaction
    "B-Drug", "I-Drug",
    "B-Disease", "I-Disease",
    "B-Symptom", "I-Symptom",
]

# Label types for evaluation
LABEL_TYPES = ["ADR", "Drug", "Disease", "Symptom"]

# ============================================================================
# Entity Normalization Configuration
# ============================================================================

# Fuzzy Matching (Approach A)
FUZZY_THRESHOLD = 80  # Minimum similarity score (0-100)
FUZZY_SCORER = "token_set_ratio"  # Options: ratio, partial_ratio, token_sort_ratio, token_set_ratio

# Embedding-Based Matching (Approach B)
EMBEDDING_THRESHOLD = 0.7  # Minimum cosine similarity (0-1)
EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding generation

# ============================================================================
# Evaluation Configuration
# ============================================================================

# Evaluation Metrics
EVAL_METRICS = ["precision", "recall", "f1", "accuracy"]

# Span Matching Strategy
# Options:
#   - "exact": Exact span match required (strict)
#   - "overlap": Partial overlap considered match (lenient)
#   - "relaxed": Overlap threshold (customizable)
SPAN_MATCH_STRATEGY = "exact"
OVERLAP_THRESHOLD = 0.5  # Used when SPAN_MATCH_STRATEGY == "relaxed"

# ============================================================================
# Performance Optimization
# ============================================================================

# GPU Configuration
USE_GPU = True  # Set to False to force CPU
GPU_ID = 0  # Specify GPU device ID if multiple GPUs available

# Caching Configuration
ENABLE_CACHING = True
CACHE_MODEL_OUTPUTS = True  # Cache model inference results

# Batch Processing
ENABLE_BATCH_PROCESSING = True
MAX_BATCH_SIZE = 32  # Maximum batch size for processing

# Parallel Processing
NUM_WORKERS = 4  # Number of parallel workers for data loading
USE_MULTIPROCESSING = True

# ============================================================================
# Reproducibility
# ============================================================================

# Random Seed
RANDOM_SEED = 42

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "cadec_ner.log"
LOG_TO_FILE = True
LOG_TO_CONSOLE = True

# ============================================================================
# Data Processing Configuration
# ============================================================================

# Text Processing
LOWERCASE_TEXT = False  # Keep original case for medical terms
NORMALIZE_WHITESPACE = True

# Entity Filtering
MIN_ENTITY_LENGTH = 1  # Minimum character length for entities
MAX_ENTITY_LENGTH = 100  # Maximum character length for entities

# ============================================================================
# Visualization Configuration
# ============================================================================

# Plot Settings
FIGURE_SIZE = (12, 8)
DPI = 300  # Resolution for saved figures
COLOR_PALETTE = "Set2"  # Seaborn color palette
SAVE_FIGURES = True
FIGURE_FORMAT = "png"  # Options: png, pdf, svg

# ============================================================================
# Validation Functions
# ============================================================================


def validate_paths() -> None:
    """Validate that all required paths exist."""
    required_dirs = [BASE_DIR, TEXT_DIR, ORIGINAL_DIR]
    missing = [d for d in required_dirs if not d.exists()]
    if missing:
        raise FileNotFoundError(
            f"Required directories not found: {', '.join(str(d) for d in missing)}"
        )


def get_device() -> str:
    """Get the computation device (cuda or cpu) based on configuration."""
    import torch
    
    if USE_GPU and torch.cuda.is_available():
        return f"cuda:{GPU_ID}"
    return "cpu"


# Validate paths on import
try:
    validate_paths()
except FileNotFoundError as e:
    import warnings
    warnings.warn(f"Path validation warning: {e}", UserWarning)

