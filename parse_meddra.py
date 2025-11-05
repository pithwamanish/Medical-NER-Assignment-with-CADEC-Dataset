"""
MedDRA annotation parser.

This module provides functionality to parse MedDRA (Medical Dictionary for
Regulatory Activities) annotation files from the CADEC dataset.

Note: For production use, consider using src.data_loader.annotation_parser
which provides enhanced error handling and logging.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_meddra_annotations(meddra_dir: Path) -> Tuple[Dict[str, List[Dict]], int, int]:
    """
    Parse MedDRA annotations from directory.
    
    MedDRA (Medical Dictionary for Regulatory Activities) is a standardized
    medical terminology used for adverse event reporting. This function
    parses all .ann files in the meddra directory.
    
    Args:
        meddra_dir: Path to directory containing MedDRA .ann files
        
    Returns:
        Tuple containing:
        - Dictionary mapping filename to list of entity dictionaries
        - Number of files processed
        - Total number of entities found
        
    Raises:
        FileNotFoundError: If meddra_dir does not exist
        ValueError: If file format is invalid
        
    Example:
        >>> from pathlib import Path
        >>> meddra_path = Path("cadec/meddra")
        >>> annotations, files, entities = parse_meddra_annotations(meddra_path)
        >>> print(f"Processed {files} files, found {entities} entities")
    """
    meddra_path = Path(meddra_dir)
    if not meddra_path.exists():
        raise FileNotFoundError(f"MedDRA directory not found: {meddra_path}")
    
    annotations: Dict[str, List[Dict]] = {}
    total_entities = 0
    files_processed = 0

    # Get all .ann files in the directory
    ann_files = list(meddra_path.glob("*.ann"))
    logger.info(f"Found {len(ann_files)} MedDRA annotation files")

    for ann_file in ann_files:
        filename = ann_file.name
        annotations[filename] = []
        files_processed += 1

        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split("\t")
                    if len(parts) != 3:
                        logger.warning(
                            f"Malformed line {line_num} in {ann_file}: "
                            f"expected 3 parts, got {len(parts)}"
                        )
                        continue

                    identifier = parts[0]
                    meddra_part = parts[1].split()
                    if len(meddra_part) != 3:
                        logger.warning(
                            f"Invalid MedDRA format at line {line_num} in {ann_file}"
                        )
                        continue

                    try:
                        meddra_code = meddra_part[0]
                        start = int(meddra_part[1])
                        end = int(meddra_part[2])
                        entity_text = parts[2]

                        entity = {
                            "identifier": identifier,
                            "meddra_code": meddra_code,
                            "start": start,
                            "end": end,
                            "text": entity_text,
                            "file_name": filename,
                        }

                        annotations[filename].append(entity)
                        total_entities += 1
                    except (ValueError, IndexError) as e:
                        logger.warning(
                            f"Error parsing line {line_num} in {ann_file}: {e}"
                        )
                        continue
        
        except Exception as e:
            logger.error(f"Error processing {ann_file}: {e}")
            continue

    logger.info(
        f"Processed {files_processed} files, found {total_entities} MedDRA entities"
    )
    return annotations, files_processed, total_entities

if __name__ == "__main__":
    meddra_dir = 'cadec/meddra'
    annotations, files_processed, total_entities = parse_meddra_annotations(meddra_dir)

    print(f"Files processed: {files_processed}")
    print(f"Total ADR entities found: {total_entities}")
    print("\nSample of parsed data structure (first file with entities):")

    # Find first file with entities for sample
    for filename, entities in annotations.items():
        if entities:
            print(f"File: {filename}")
            for entity in entities[:3]:  # Show first 3 entities
                print(f"  {entity}")
            break