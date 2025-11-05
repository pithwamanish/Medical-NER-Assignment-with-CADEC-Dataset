# CADEC NER Project

Medical Named Entity Recognition (NER) project using the CADEC (Consumer Adverse Drug Event) dataset.

## Project Overview

This project implements a comprehensive pipeline for medical Named Entity Recognition, focusing on extracting and evaluating medical entities from patient-reported adverse drug events. The project follows best practices for code organization, documentation, testing, and reproducibility.

## Project Structure

```
.
├── config.py                 # Centralized configuration
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── data_loader/          # Data loading and parsing
│   │   ├── __init__.py
│   │   ├── annotation_parser.py   # Parse CADEC annotations
│   │   └── text_loader.py          # Load text files
│   │
│   ├── model_inference/      # Model inference
│   │   ├── __init__.py
│   │   ├── ner_model.py            # NER model wrapper
│   │   └── embedding_model.py      # Embedding model for similarity
│   │
│   ├── evaluation/           # Evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py              # Precision, recall, F1, etc.
│   │
│   ├── visualization/        # Plotting utilities
│   │   ├── __init__.py
│   │   └── plots.py                # Visualization functions
│   │
│   └── utils/                # Utilities
│       ├── __init__.py
│       └── logging_utils.py         # Logging setup
│
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_annotation_parser.py
│   └── test_metrics.py
│
├── cadec/                    # CADEC dataset
│   ├── text/                 # Text files (.txt)
│   ├── original/             # Original annotations (.ann)
│   ├── sct/                  # SNOMED CT annotations
│   └── meddra/               # MedDRA annotations
│
├── outputs/                  # Generated outputs
│   ├── models/               # Cached models
│   ├── results/              # Evaluation results
│   ├── cache/                # Cache files
│   └── logs/                 # Log files
│
└── task*.ipynb               # Jupyter notebooks for tasks
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster inference

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Medical NER Assignment with CADEC Dataset"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import config; print('Installation successful!')"
   ```

## Quick Start

### Basic Usage

```python
# Setup logging
from src.utils.logging_utils import setup_logging
setup_logging()

# Load data
from src.data_loader.annotation_parser import load_ground_truth
from pathlib import Path

ann_file = Path("cadec/original/ARTHROTEC.1.ann")
entities = load_ground_truth(ann_file)
print(f"Loaded {len(entities)} entities")

# Run NER model
from src.model_inference import load_ner_model

model = load_ner_model()
text = "I felt drowsy after taking aspirin."
predictions = model.predict(text)
print(predictions)

# Evaluate predictions
from src.evaluation.metrics import calculate_metrics, ClassificationReport

metrics = calculate_metrics(predictions, entities)
report = ClassificationReport(metrics)
print(report)
```

## Configuration

All configuration is centralized in `config.py`. Key settings include:

### Model Selection

- **LLM Model**: `config.LLM_MODEL_NAME`
  - Default: `"dmis-lab/biobert-base-cased-v1.1"` (BioBERT for medical NER)
  - Alternative: `"blaze999/Medical-NER"` (domain-specific)

- **Embedding Model**: `config.EMBEDDING_MODEL_NAME`
  - Default: `"sentence-transformers/all-MiniLM-L6-v2"` (fast)
  - Alternative: `"dmis-lab/biobert-base-cased-v1.1"` (medical domain)

### Paths

- Dataset paths: `config.BASE_DIR`, `config.TEXT_DIR`, etc.
- Output paths: `config.OUTPUT_DIR`, `config.RESULTS_DIR`, etc.

### Hyperparameters

- Batch size: `config.BATCH_SIZE` (default: 16)
- Sequence length: `config.MAX_SEQUENCE_LENGTH` (default: 512)
- Learning rate: `config.LEARNING_RATE` (default: 2e-5)

### Evaluation

- Matching strategy: `config.SPAN_MATCH_STRATEGY`
  - `"exact"`: Exact span match (strict)
  - `"overlap"`: Any overlap counts (lenient)
  - `"relaxed"`: Overlap threshold-based

## Modules Overview

### Data Loading (`src/data_loader/`)

**`annotation_parser.py`**: Parses CADEC annotation formats
- `load_ground_truth()`: Load original annotations
- `parse_snomed_annotations()`: Parse SNOMED CT annotations
- `parse_meddra_annotations()`: Parse MedDRA annotations
- `AnnotationParser`: Unified parser class

**`text_loader.py`**: Loads text files
- `load_text_file()`: Load single text file
- `load_all_texts()`: Load all texts from directory

### Model Inference (`src/model_inference/`)

**`ner_model.py`**: NER model wrapper
- `NERModel`: Class for token classification models
- `load_ner_model()`: Convenience function to load models
- Supports BioBERT, Medical-NER, and other transformer models

**`embedding_model.py`**: Embedding model for similarity
- `EmbeddingModel`: Class for generating embeddings
- `load_embedding_model()`: Convenience function
- Used for entity normalization via semantic similarity

### Evaluation (`src/evaluation/`)

**`metrics.py`**: Evaluation metrics
- `calculate_metrics()`: Comprehensive metrics calculation
- `span_match()`: Span matching strategies
- `precision_recall_f1()`: Core metrics
- `confusion_matrix()`: Confusion matrix calculation
- `ClassificationReport`: Structured report generator

### Visualization (`src/visualization/`)

**`plots.py`**: Plotting utilities
- `plot_metrics()`: Plot precision/recall/F1 by label
- `plot_confusion_matrix()`: Confusion matrix heatmap
- `plot_entity_distribution()`: Entity label distribution
- `plot_performance_comparison()`: Compare multiple runs/models

## Testing

Run unit tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_annotation_parser.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Code Quality Standards

### Type Hints

All functions include type hints for parameters and return values:

```python
def load_ground_truth(ann_file_path: Path) -> List[Dict]:
    ...
```

### Docstrings

All functions use Google-style docstrings:

```python
def function_name(param: type) -> return_type:
    """
    Brief description.
    
    Longer description explaining what the function does,
    its parameters, return values, and any exceptions.
    
    Args:
        param: Parameter description
        
    Returns:
        Return value description
        
    Raises:
        ExceptionType: When this exception is raised
        
    Example:
        >>> result = function_name(value)
        >>> print(result)
    """
```

### Error Handling

Functions include try-except blocks with appropriate logging:

```python
try:
    # Operation
except SpecificException as e:
    logger.error(f"Error message: {e}")
    raise
```

### Resource Management

Files are properly opened and closed using context managers:

```python
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
```

## Performance Optimization

### Batch Processing

Models support batch processing for efficiency:

```python
model = load_ner_model()
texts = ["text1", "text2", "text3"]
results = model.predict_batch(texts, batch_size=16)
```

### Caching

Model outputs can be cached to avoid recomputation:

```python
# Enable caching in config.py
config.ENABLE_CACHING = True
config.CACHE_MODEL_OUTPUTS = True
```

### GPU Utilization

GPU is automatically used when available:

```python
# Check device
device = config.get_device()  # Returns "cuda:0" or "cpu"
```

## Reproducibility

### Random Seeds

Set random seeds for reproducibility:

```python
import random
import numpy as np
import torch

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)
```

### Version Pinning

Dependencies are pinned in `requirements.txt` for reproducibility.

## Logging

Logging is configured via `src/utils/logging_utils.py`:

```python
from src.utils.logging_utils import setup_logging

setup_logging()  # Uses config.py settings
```

Logs are written to both console and file (configurable).

## Medical Terminology

Key medical terminologies used in this project:

- **ADR (Adverse Drug Reaction)**: Harmful or unintended response to medication
- **SNOMED CT**: Systematized Nomenclature of Medicine -- Clinical Terms
- **MedDRA**: Medical Dictionary for Regulatory Activities
- **BIO Tagging**: Begin-Inside-Outside tagging scheme for NER

## References

- **CADEC Dataset**: Karimi, S., et al. (2015). "Cadec: A corpus of adverse drug event annotations"
- **BioBERT**: Lee, J., et al. (2019). "BioBERT: a pre-trained biomedical language representation model"
- **Sentence Transformers**: Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

## Contributing

When adding new features:

1. Follow the existing module structure
2. Add type hints and docstrings
3. Write unit tests
4. Update this README if needed
5. Ensure code passes linting

## License

[Specify your license here]

## Contact

[Your contact information]

## Running in Google Colab

See **[COLAB_SETUP.md](COLAB_SETUP.md)** for detailed instructions on running this project in Google Colab.

**Quick Start for Colab:**
1. Upload project to Google Drive or clone via Git
2. Open `colab_setup.ipynb` in Colab
3. Run all cells to set up the environment
4. Start using the modules!

# Medical-NER-Assignment-with-CADEC-Dataset
