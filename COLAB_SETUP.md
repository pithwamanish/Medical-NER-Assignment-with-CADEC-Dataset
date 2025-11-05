# Running CADEC NER Project in Google Colab

This guide explains how to set up and run the CADEC NER project in Google Colab.

## Quick Start

### Option 1: Using Git (Recommended - No Drive Needed!)

If you're already using `git pull` in Colab, this is the simplest method:

```python
# Navigate to your repository (if already cloned)
import os
import sys
os.chdir('/content/your-repo-name')  # Update with your repo path

# Pull latest changes (if needed)
!git pull

# Install dependencies
!pip install -q -r requirements.txt

# Add to Python path
sys.path.insert(0, os.getcwd())

# Test imports
import config
from src.utils.logging_utils import setup_logging
setup_logging()
print("✅ Setup complete!")
```

**First time cloning?**
```python
# Clone repository
!git clone https://github.com/yourusername/your-repo.git
%cd your-repo

# Install dependencies
!pip install -q -r requirements.txt

# Add to Python path
import sys
sys.path.insert(0, os.getcwd())
```

### Option 2: Upload Project to Google Drive (Alternative)

Only use this if you're NOT using Git:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your project directory
import os
os.chdir('/content/drive/MyDrive/YourProjectPath')  # Update this path

# Install dependencies
!pip install -r requirements.txt

# Verify setup
!python setup_project.py
```

### Option 3: Upload Directly in Colab

1. Click on the folder icon in Colab sidebar
2. Upload your project files using the upload button
3. Or drag and drop the entire folder

## Complete Setup Code for Colab

### Using Git (Recommended)

Copy and run this in a Colab cell:

```python
# ============================================================================
# CADEC NER Project - Google Colab Setup (Git Method)
# ============================================================================

import sys
import os
from pathlib import Path

# Step 1: Clone or navigate to repository
# If first time, clone:
# !git clone https://github.com/yourusername/your-repo.git
# %cd your-repo

# If already cloned, just navigate:
# %cd your-repo

# Or if using git pull (as you mentioned):
REPO_PATH = '/content/your-repo-name'  # Update with your repo name/path
if not os.path.exists(REPO_PATH):
    # Clone if doesn't exist
    !git clone https://github.com/yourusername/your-repo.git
else:
    # Pull latest changes
    os.chdir(REPO_PATH)
    !git pull

os.chdir(REPO_PATH)
print(f"✅ Current directory: {os.getcwd()}")

# Step 2: Install dependencies
print("\nInstalling dependencies...")
!pip install -q -r requirements.txt

# Step 3: Add to Python path
sys.path.insert(0, os.getcwd())
print(f"✅ Python path updated")

# Step 4: Test imports
print("\nTesting imports...")
try:
    import config
    from src.utils.logging_utils import setup_logging
    from src.data_loader import load_ground_truth
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Setup logging
setup_logging()
print("\n✅ Setup complete! Ready to use.")
```

### Using Google Drive (Alternative)

Only use this if NOT using Git:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project directory
PROJECT_PATH = '/content/drive/MyDrive/Medical NER Assignment with CADEC Dataset'
os.chdir(PROJECT_PATH)
sys.path.insert(0, PROJECT_PATH)

# Install dependencies
!pip install -q -r requirements.txt
```

## Using the Modules in Colab

### Basic Usage Example

```python
# Import modules
from src.utils.logging_utils import setup_logging
from src.data_loader.annotation_parser import load_ground_truth
from src.model_inference import load_ner_model
from src.evaluation.metrics import calculate_metrics, ClassificationReport
from pathlib import Path

# Setup logging
setup_logging()

# Load data
ann_file = Path("cadec/original/ARTHROTEC.1.ann")
entities = load_ground_truth(ann_file)
print(f"Loaded {len(entities)} entities")

# Run NER model (will download on first use)
print("Loading NER model (this may take a few minutes on first run)...")
model = load_ner_model()
text = "I felt drowsy after taking aspirin."
predictions = model.predict(text)
print(f"Predictions: {predictions}")
```

## Colab-Specific Considerations

### 1. GPU Usage

Colab provides free GPU access. To enable:

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

The project automatically uses GPU if available via `config.get_device()`.

### 2. File Paths

In Colab, use absolute paths or ensure you're in the correct directory:

```python
from pathlib import Path
import config

# Paths are relative to project root
print(f"Base directory: {config.BASE_DIR}")
print(f"Text directory: {config.TEXT_DIR}")

# Verify paths exist
assert config.TEXT_DIR.exists(), "Text directory not found!"
```

### 3. Dataset Upload

If your dataset is large, consider:

**Option A: Upload to Google Drive**
```python
# Mount drive and link to dataset
from google.colab import drive
drive.mount('/content/drive')

# Create symlink or copy dataset
import shutil
source = '/content/drive/MyDrive/cadec'
destination = '/content/cadec'
shutil.copytree(source, destination, dirs_exist_ok=True)
```

**Option B: Use Colab's file upload**
```python
# Upload files directly
from google.colab import files
uploaded = files.upload()  # Interactive upload
```

### 4. Model Caching

Models are cached in `outputs/models/`. To persist across sessions:

```python
# Copy models to Drive
import shutil
shutil.copytree(
    'outputs/models',
    '/content/drive/MyDrive/cadec_models',
    dirs_exist_ok=True
)

# Later, restore from Drive
shutil.copytree(
    '/content/drive/MyDrive/cadec_models',
    'outputs/models',
    dirs_exist_ok=True
)
```

### 5. Memory Management

Colab has limited RAM. For large datasets:

```python
# Process in batches
from src.data_loader import load_all_texts

texts = load_all_texts()
batch_size = 10

for i in range(0, len(texts), batch_size):
    batch = dict(list(texts.items())[i:i+batch_size])
    # Process batch
    # Clear memory
    del batch
    import gc
    gc.collect()
```

## Example: Complete Colab Notebook

Create a new Colab notebook with this structure:

```python
# Cell 1: Setup
from google.colab import drive
import os
drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/YourProjectPath')
!pip install -q -r requirements.txt

# Cell 2: Imports
import sys
sys.path.insert(0, os.getcwd())
from src.utils.logging_utils import setup_logging
setup_logging()

# Cell 3: Load data
from src.data_loader import load_ground_truth
from pathlib import Path
ann_file = Path("cadec/original/ARTHROTEC.1.ann")
entities = load_ground_truth(ann_file)

# Cell 4: Run model
from src.model_inference import load_ner_model
model = load_ner_model()
text = "I felt drowsy after taking aspirin."
predictions = model.predict(text)

# Cell 5: Evaluate
from src.evaluation import calculate_metrics
metrics = calculate_metrics(predictions, entities)
print(metrics)
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:**
```python
import sys
sys.path.insert(0, '/content/your-project-path')
```

### Issue: Dataset not found

**Solution:**
```python
from pathlib import Path
import os

# Check current directory
print(f"Current dir: {os.getcwd()}")

# Check if dataset exists
dataset_path = Path("cadec")
if not dataset_path.exists():
    print("Dataset not found. Upload it first!")
else:
    print(f"Dataset found: {dataset_path}")
```

### Issue: GPU not available

**Solution:**
- Go to Runtime > Change runtime type
- Set Hardware accelerator to "GPU"
- Save and restart runtime

### Issue: Out of memory

**Solution:**
```python
# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Process in smaller batches
# Reduce batch_size in config.py
```

## Saving Your Work

### Save outputs to Drive

```python
import shutil
from datetime import datetime

# Create timestamped backup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f'/content/drive/MyDrive/backups/{timestamp}'

shutil.copytree('outputs', f'{backup_path}/outputs')
print(f"Backed up to: {backup_path}")
```

### Download results

```python
from google.colab import files

# Download specific file
files.download('outputs/results/metrics_plot.png')

# Or download entire results directory
!zip -r results.zip outputs/results/
files.download('results.zip')
```

## Performance Tips for Colab

1. **Use GPU**: Always enable GPU for model inference
2. **Batch Processing**: Process data in batches to avoid memory issues
3. **Cache Models**: Download models once and reuse them
4. **Save Progress**: Regularly save intermediate results to Drive
5. **Monitor Resources**: Check Colab's resource usage (RAM, disk)

## Quick Reference

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Change directory
import os
os.chdir('/path/to/project')

# Install packages
!pip install package_name

# Check GPU
import torch
print(torch.cuda.is_available())

# Clear memory
import gc
gc.collect()
torch.cuda.empty_cache()
```

## Next Steps

1. Upload your project to Google Drive
2. Open Colab and mount Drive
3. Run the setup code
4. Start using the modules in your notebooks!

For more details, see the main README.md file.

