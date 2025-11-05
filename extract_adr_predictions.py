"""
Example script for extracting and evaluating ADR predictions.

This is a demonstration script showing how to:
1. Load ground truth annotations
2. Compare predictions with gold standard
3. Calculate TP, FP, FN metrics

For production use, consider using the modules in src/ directory
which provide more robust functionality.
"""

import logging
from pathlib import Path
from typing import List, Dict, Set

# Import from src module (preferred approach)
try:
    from src.data_loader.annotation_parser import load_ground_truth
    from src.evaluation.metrics import calculate_metrics, ClassificationReport
except ImportError:
    # Fallback for standalone execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from src.data_loader.annotation_parser import load_ground_truth
    from src.evaluation.metrics import calculate_metrics, ClassificationReport

import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_ground_truth_legacy(ann_file_path: Path) -> List[Dict]:
    """
    Load and parse ground truth annotation file from 'original' subdirectory.

    Format: TAG\tLABEL START END\tTEXT
    Example: T1	ADR 9 19	bit drowsy
    Example with multiple ranges: T6	Symptom 66 74;76 94;98 107	the heel I couldn't walk on very well

    Parameters:
    -----------
    ann_file_path : Path
        Path to the .ann annotation file

    Returns:
    --------
    List[Dict]
        List of entity dictionaries with:
        - 'label': Entity type (ADR, Drug, Disease, Symptom)
        - 'text': Entity text
        - 'start': Start character position
        - 'end': End character position
        - 'tag': Original tag identifier (T1, T2, etc.)
    """

    entities = []

    try:
        with open(ann_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comment lines (starting with '#')
                if not line or line.startswith('#'):
                    continue

                # Parse entity annotation lines (starting with 'T' followed by a number)
                # Format: TAG\tLABEL RANGES\tTEXT
                match = re.match(r'^(T\d+)\t([^\t]+)\t(.+)$', line)
                if match:
                    tag = match.group(1)
                    label_and_ranges = match.group(2)
                    text = match.group(3)

                    # Extract label type (first word) and ranges (remaining part)
                    parts = label_and_ranges.split(None, 1)
                    if len(parts) < 2:
                        continue

                    label_type = parts[0]
                    ranges_str = parts[1]

                    # Only process ADR, Drug, Disease, Symptom labels
                    if label_type not in LABEL_TYPES:
                        continue

                    # Extract ranges (can be multiple pairs separated by semicolons)
                    ranges = []
                    if ';' in ranges_str:
                        # Multiple ranges format: "START1 END1;START2 END2;..."
                        range_pairs = ranges_str.split(';')
                        for rp in range_pairs:
                            rp = rp.strip()
                            if rp:
                                range_nums = rp.split()
                                if len(range_nums) >= 2:
                                    try:
                                        start = int(range_nums[0])
                                        end = int(range_nums[1])
                                        ranges.append((start, end))
                                    except ValueError:
                                        continue
                    else:
                        # Single range format: "START END"
                        range_nums = ranges_str.split()
                        if len(range_nums) >= 2:
                            try:
                                start = int(range_nums[0])
                                end = int(range_nums[1])
                                ranges = [(start, end)]
                            except ValueError:
                                continue

                    # Create entity entries for each range
                    # For multiple ranges, we create separate entities (standard practice in NER)
                    for start, end in ranges:
                        entities.append({
                            'label': label_type,
                            'text': text.strip(),
                            'start': start,
                            'end': end,
                            'tag': tag
                        })

    except Exception as e:
        print(f"Error loading ground truth from {ann_file_path}: {e}")
        return []

    return entities

def evaluate_adr_predictions(
    text_dir: Path = None,
    original_dir: Path = None,
    pred_entities_func = None,
) -> Dict:
    """
    Evaluate ADR predictions against ground truth.
    
    Args:
        text_dir: Directory containing text files. If None, uses config.TEXT_DIR
        original_dir: Directory containing original annotations. If None, uses
                     config.ORIGINAL_DIR
        pred_entities_func: Function that takes text_file Path and returns
                           list of predicted entities. If None, uses dummy
                           predictions (first half of ground truth)
    
    Returns:
        Dictionary containing evaluation results and metrics
    """
    text_dir = text_dir or config.TEXT_DIR
    original_dir = original_dir or config.ORIGINAL_DIR
    
    text_files = list(text_dir.glob("*.txt"))
    results = {}
    all_tp_samples = []
    all_fp_samples = []
    all_fn_samples = []
    
    logger.info(f"Processing {len(text_files)} files")
    
    for text_file in text_files:
        # Load ground truth
        gt_file = original_dir / text_file.name.replace(".txt", ".ann")
        if not gt_file.exists():
            logger.warning(f"Ground truth file not found: {gt_file}")
            continue
        
        try:
            gt_entities = load_ground_truth(gt_file)
            
            # Filter to ADR only
            gt_adr = [e for e in gt_entities if e["label"] == "ADR"]
            
            # Get predictions
            if pred_entities_func:
                pred_entities = pred_entities_func(text_file)
                pred_adr = [e for e in pred_entities if e.get("label") == "ADR"]
            else:
                # Dummy: use first half of ground truth as predictions
                pred_adr = gt_adr[:len(gt_adr)//2] if len(gt_adr) > 1 else []
            
            # Calculate metrics using src.evaluation module
            metrics = calculate_metrics(pred_adr, gt_adr, label_types=["ADR"])
            
            # Store results
            results[text_file.name] = {
                "metrics": metrics,
                "gt_count": len(gt_adr),
                "pred_count": len(pred_adr),
            }
            
        except Exception as e:
            logger.error(f"Error processing {text_file}: {e}")
            continue
    
    # Aggregate overall metrics
    overall_tp = sum(r["metrics"]["ADR"]["tp"] for r in results.values())
    overall_fp = sum(r["metrics"]["ADR"]["fp"] for r in results.values())
    overall_fn = sum(r["metrics"]["ADR"]["fn"] for r in results.values())
    
    return {
        "results": results,
        "overall": {
            "tp": overall_tp,
            "fp": overall_fp,
            "fn": overall_fn,
            "files_processed": len(results),
        }
    }


if __name__ == "__main__":
    # Example usage
    evaluation_results = evaluate_adr_predictions()
    
    print("\n" + "=" * 60)
    print("ADR Evaluation Summary")
    print("=" * 60)
    print(f"Files processed: {evaluation_results['overall']['files_processed']}")
    print(f"Total True Positives (TP): {evaluation_results['overall']['tp']}")
    print(f"Total False Positives (FP): {evaluation_results['overall']['fp']}")
    print(f"Total False Negatives (FN): {evaluation_results['overall']['fn']}")
    print("=" * 60)