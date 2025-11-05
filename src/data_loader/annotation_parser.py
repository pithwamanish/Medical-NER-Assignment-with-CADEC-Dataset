"""
Annotation parser for CADEC dataset annotation files.

This module handles parsing of various annotation formats:
- Original annotations (.ann files with BIO tags)
- SNOMED CT annotations (SCT subdirectory)
- MedDRA annotations (MedDRA subdirectory)
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import config

logger = logging.getLogger(__name__)


class AnnotationParser:
    """
    Unified parser for CADEC annotation files.
    
    This class provides methods to parse different annotation formats
    used in the CADEC dataset, including handling of multiple ranges,
    SNOMED codes, and MedDRA codes.
    """
    
    def __init__(self, label_types: Optional[List[str]] = None):
        """
        Initialize the annotation parser.
        
        Args:
            label_types: List of label types to process. If None, uses
                        config.LABEL_TYPES
        """
        self.label_types = label_types or config.LABEL_TYPES
        logger.debug(f"Initialized AnnotationParser with labels: {self.label_types}")
    
    def parse_entity_line(
        self, 
        line: str, 
        file_name: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Parse a single annotation line.
        
        Format: TAG\tLABEL RANGES\tTEXT
        Example: T1\tADR 9 19\tbit drowsy
        Example with multiple ranges: T6\tSymptom 66 74;76 94;98 107\ttext
        
        Args:
            line: Annotation line to parse
            file_name: Optional source file name for tracking
            
        Returns:
            Dictionary with entity information, or None if parsing fails
        """
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            return None
        
        # Parse entity annotation lines (starting with 'T' followed by a number)
        match = re.match(r"^(T\d+)\t([^\t]+)\t(.+)$", line)
        if not match:
            return None
        
        tag = match.group(1)
        label_and_ranges = match.group(2)
        text = match.group(3)
        
        # Extract label type (first word) and ranges (remaining part)
        parts = label_and_ranges.split(None, 1)
        if len(parts) < 2:
            return None
        
        label_type = parts[0]
        ranges_str = parts[1]
        
        # Filter by label types
        if label_type not in self.label_types:
            return None
        
        # Extract ranges (can be multiple pairs separated by semicolons)
        ranges = self._parse_ranges(ranges_str)
        if not ranges:
            return None
        
        # Create entity entries for each range
        entities = []
        for start, end in ranges:
            entities.append({
                "label": label_type,
                "text": text.strip(),
                "start": start,
                "end": end,
                "tag": tag,
                "file_name": file_name,
            })
        
        # Return single entity if one range, list if multiple
        return entities if len(entities) > 1 else entities[0]
    
    def _parse_ranges(self, ranges_str: str) -> List[Tuple[int, int]]:
        """
        Parse character ranges from string.
        
        Handles both single range ("START END") and multiple ranges
        ("START1 END1;START2 END2;...").
        
        Args:
            ranges_str: String containing range information
            
        Returns:
            List of (start, end) tuples
        """
        ranges = []
        
        if ";" in ranges_str:
            # Multiple ranges format: "START1 END1;START2 END2;..."
            range_pairs = ranges_str.split(";")
            for rp in range_pairs:
                rp = rp.strip()
                if not rp:
                    continue
                range_nums = rp.split()
                if len(range_nums) >= 2:
                    try:
                        start = int(range_nums[0])
                        end = int(range_nums[1])
                        ranges.append((start, end))
                    except ValueError:
                        logger.warning(f"Invalid range format: {rp}")
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
                    logger.warning(f"Invalid range format: {ranges_str}")
        
        return ranges


def load_ground_truth(ann_file_path: Path) -> List[Dict]:
    """
    Load and parse ground truth annotation file from 'original' subdirectory.
    
    This function parses the standard CADEC annotation format and returns
    a list of entity dictionaries. Each entity represents a medical term
    (ADR, Drug, Disease, or Symptom) with its position in the text.
    
    Format: TAG\tLABEL START END\tTEXT
    Example: T1\tADR 9 19\tbit drowsy
    Example with multiple ranges: T6\tSymptom 66 74;76 94;98 107\ttext
    
    Args:
        ann_file_path: Path to the .ann annotation file
        
    Returns:
        List of entity dictionaries with:
        - 'label': Entity type (ADR, Drug, Disease, Symptom)
        - 'text': Entity text
        - 'start': Start character position
        - 'end': End character position
        - 'tag': Original tag identifier (T1, T2, etc.)
        - 'file_name': Source file name
        
    Raises:
        FileNotFoundError: If annotation file does not exist
        ValueError: If file format is invalid
        
    Example:
        >>> from pathlib import Path
        >>> ann_file = Path("cadec/original/sample.ann")
        >>> entities = load_ground_truth(ann_file)
        >>> print(f"Found {len(entities)} entities")
    """
    if not ann_file_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file_path}")
    
    parser = AnnotationParser()
    entities = []
    
    try:
        with open(ann_file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    result = parser.parse_entity_line(line, ann_file_path.name)
                    if result is None:
                        continue
                    
                    # Handle both single entity and list of entities (multiple ranges)
                    if isinstance(result, list):
                        entities.extend(result)
                    else:
                        entities.append(result)
                except Exception as e:
                    logger.warning(
                        f"Error parsing line {line_num} in {ann_file_path}: {e}"
                    )
                    continue
        
        logger.info(f"Loaded {len(entities)} entities from {ann_file_path}")
        return entities
    
    except Exception as e:
        logger.error(f"Error loading ground truth from {ann_file_path}: {e}")
        raise ValueError(f"Failed to parse annotation file: {e}") from e


def parse_snomed_annotations(ann_file_path: Path) -> List[Dict]:
    """
    Parse SNOMED CT annotation file from 'sct' subdirectory.
    
    SNOMED CT (Systematized Nomenclature of Medicine -- Clinical Terms)
    is a comprehensive clinical terminology system. This function extracts
    SNOMED codes and their corresponding standard descriptions.
    
    Format: TAG\tCODE | DESC | START END\tENTITY_TEXT
    Example: TT1\t271782001 | Drowsy | 9 19\tbit drowsy
    Example with multiple codes: TT1\t102498003 | Agony | or 76948002|Severe pain| 260 265
    
    Args:
        ann_file_path: Path to the .ann file in sct directory
        
    Returns:
        List of SNOMED CT entity dictionaries with:
        - 'identifier': Tag identifier (TT1, TT2, etc.)
        - 'snomed_codes': List of SNOMED CT codes
        - 'snomed_descriptions': List of standard descriptions
        - 'char_start': Start character position
        - 'char_end': End character position
        - 'entity_text': Actual entity text from document
        - 'file_name': Source file name
        
    Example:
        >>> from pathlib import Path
        >>> sct_file = Path("cadec/sct/sample.ann")
        >>> entities = parse_snomed_annotations(sct_file)
        >>> print(f"Found {len(entities)} SNOMED entities")
    """
    if not ann_file_path.exists():
        raise FileNotFoundError(f"SNOMED annotation file not found: {ann_file_path}")
    
    entities = []
    
    try:
        with open(ann_file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                # Parse format: TAG\tCODE_INFO | START END\tENTITY_TEXT
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                
                identifier = parts[0]
                code_and_desc_str = parts[1]
                entity_text = parts[2]
                
                # Find character ranges (last two numbers in the string)
                range_match = re.search(r"\s+(\d+)\s+(\d+)\s*$", code_and_desc_str)
                if not range_match:
                    continue
                
                char_start = int(range_match.group(1))
                char_end = int(range_match.group(2))
                
                # Extract code and description part (before the ranges)
                code_desc_part = code_and_desc_str[:range_match.start()].strip()
                
                # Parse SNOMED codes and descriptions
                snomed_codes = []
                snomed_descriptions = []
                
                # Handle multiple code pairs separated by "or"
                code_parts = code_desc_part.split(" or ")
                for code_part in code_parts:
                    code_part = code_part.strip()
                    if "|" in code_part:
                        # Format: "CODE | DESC"
                        code_desc = [x.strip() for x in code_part.split("|")]
                        if len(code_desc) >= 2:
                            snomed_codes.append(code_desc[0])
                            snomed_descriptions.append(code_desc[1])
                
                if snomed_codes:  # Only add if we found at least one code
                    entities.append({
                        "identifier": identifier,
                        "snomed_codes": snomed_codes,
                        "snomed_descriptions": snomed_descriptions,
                        "char_start": char_start,
                        "char_end": char_end,
                        "entity_text": entity_text,
                        "file_name": ann_file_path.name,
                    })
        
        logger.info(f"Parsed {len(entities)} SNOMED entities from {ann_file_path}")
        return entities
    
    except Exception as e:
        logger.error(f"Error parsing SNOMED annotations from {ann_file_path}: {e}")
        raise ValueError(f"Failed to parse SNOMED annotation file: {e}") from e


def parse_meddra_annotations(meddra_dir: Path) -> Tuple[Dict[str, List[Dict]], int, int]:
    """
    Parse MedDRA annotations from directory.
    
    MedDRA (Medical Dictionary for Regulatory Activities) is a standardized
    medical terminology used for adverse event reporting. This function
    parses all .ann files in the meddra directory.
    
    Format: IDENTIFIER\tMEDDRA_CODE START END\tENTITY_TEXT
    Example: T1\t10012345 9 19\tbit drowsy
    
    Args:
        meddra_dir: Directory containing MedDRA .ann files
        
    Returns:
        Tuple of:
        - Dictionary mapping filename to list of entities
        - Number of files processed
        - Total number of entities found
        
    Example:
        >>> from pathlib import Path
        >>> meddra_path = Path("cadec/meddra")
        >>> annotations, files, entities = parse_meddra_annotations(meddra_path)
        >>> print(f"Processed {files} files, found {entities} entities")
    """
    if not meddra_dir.exists():
        raise FileNotFoundError(f"MedDRA directory not found: {meddra_dir}")
    
    annotations: Dict[str, List[Dict]] = {}
    total_entities = 0
    files_processed = 0
    
    ann_files = list(meddra_dir.glob("*.ann"))
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
                        continue  # Skip malformed lines
                    
                    identifier = parts[0]
                    meddra_part = parts[1].split()
                    if len(meddra_part) != 3:
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

