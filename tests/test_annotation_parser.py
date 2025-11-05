"""
Unit tests for annotation parsing functions.

Tests the annotation parser module for handling various
CADEC annotation formats including original, SNOMED, and MedDRA.
"""

import pytest
from pathlib import Path
from typing import List, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader.annotation_parser import (
    AnnotationParser,
    load_ground_truth,
    parse_snomed_annotations,
    parse_meddra_annotations,
)


class TestAnnotationParser:
    """Test suite for AnnotationParser class."""
    
    def test_parser_init(self):
        """Test parser initialization."""
        parser = AnnotationParser()
        assert parser.label_types is not None
        assert len(parser.label_types) > 0
    
    def test_parse_single_range(self):
        """Test parsing entity with single character range."""
        parser = AnnotationParser()
        line = "T1\tADR 10 20\tdrowsy feeling"
        result = parser.parse_entity_line(line)
        
        assert result is not None
        assert result["label"] == "ADR"
        assert result["start"] == 10
        assert result["end"] == 20
        assert result["text"] == "drowsy feeling"
        assert result["tag"] == "T1"
    
    def test_parse_multiple_ranges(self):
        """Test parsing entity with multiple character ranges."""
        parser = AnnotationParser()
        line = "T6\tSymptom 66 74;76 94;98 107\tthe heel I couldn't walk"
        result = parser.parse_entity_line(line)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(r["label"] == "Symptom" for r in result)
        assert all(r["text"] == "the heel I couldn't walk" for r in result)
    
    def test_parse_empty_line(self):
        """Test parsing empty or comment lines."""
        parser = AnnotationParser()
        assert parser.parse_entity_line("") is None
        assert parser.parse_entity_line("# Comment line") is None
        assert parser.parse_entity_line("   ") is None
    
    def test_parse_invalid_format(self):
        """Test parsing invalid annotation format."""
        parser = AnnotationParser()
        assert parser.parse_entity_line("Invalid format") is None
        assert parser.parse_entity_line("T1\tADR") is None  # Missing ranges and text
    
    def test_parse_label_filtering(self):
        """Test that only configured labels are parsed."""
        parser = AnnotationParser(label_types=["ADR", "Drug"])
        line = "T1\tDisease 10 20\tsome disease"
        assert parser.parse_entity_line(line) is None  # Disease not in filter


class TestGroundTruthLoader:
    """Test suite for ground truth loading."""
    
    def test_load_ground_truth_format(self, tmp_path):
        """Test loading ground truth from file."""
        # Create temporary annotation file
        ann_file = tmp_path / "test.ann"
        ann_file.write_text(
            "T1\tADR 10 20\tdrowsy feeling\n"
            "T2\tDrug 25 32\taspirin\n"
            "# Comment line\n"
        )
        
        entities = load_ground_truth(ann_file)
        
        assert len(entities) == 2
        assert entities[0]["label"] == "ADR"
        assert entities[1]["label"] == "Drug"
    
    def test_load_ground_truth_missing_file(self):
        """Test error handling for missing file."""
        missing_file = Path("nonexistent_file.ann")
        with pytest.raises(FileNotFoundError):
            load_ground_truth(missing_file)


class TestSNOMParser:
    """Test suite for SNOMED annotation parsing."""
    
    def test_parse_snomed_single_code(self, tmp_path):
        """Test parsing SNOMED annotation with single code."""
        ann_file = tmp_path / "test_sct.ann"
        ann_file.write_text(
            "TT1\t271782001 | Drowsy | 9 19\tbit drowsy\n"
        )
        
        entities = parse_snomed_annotations(ann_file)
        
        assert len(entities) == 1
        assert entities[0]["snomed_codes"] == ["271782001"]
        assert entities[0]["snomed_descriptions"] == ["Drowsy"]
        assert entities[0]["char_start"] == 9
        assert entities[0]["char_end"] == 19
    
    def test_parse_snomed_multiple_codes(self, tmp_path):
        """Test parsing SNOMED annotation with multiple codes."""
        ann_file = tmp_path / "test_sct.ann"
        ann_file.write_text(
            "TT1\t102498003 | Agony | or 76948002|Severe pain| 260 265\tsevere agony\n"
        )
        
        entities = parse_snomed_annotations(ann_file)
        
        assert len(entities) == 1
        assert len(entities[0]["snomed_codes"]) == 2
        assert "102498003" in entities[0]["snomed_codes"]
        assert "76948002" in entities[0]["snomed_codes"]
    
    def test_parse_snomed_missing_file(self):
        """Test error handling for missing SNOMED file."""
        missing_file = Path("nonexistent_sct.ann")
        with pytest.raises(FileNotFoundError):
            parse_snomed_annotations(missing_file)


class TestMedDRAParser:
    """Test suite for MedDRA annotation parsing."""
    
    def test_parse_meddra_format(self, tmp_path):
        """Test parsing MedDRA annotations."""
        meddra_dir = tmp_path / "meddra"
        meddra_dir.mkdir()
        
        ann_file = meddra_dir / "test.ann"
        ann_file.write_text(
            "T1\t10012345 10 20\tdrowsy feeling\n"
            "T2\t10023456 25 32\taspirin\n"
        )
        
        annotations, files_processed, total_entities = parse_meddra_annotations(meddra_dir)
        
        assert files_processed == 1
        assert total_entities == 2
        assert "test.ann" in annotations
        assert len(annotations["test.ann"]) == 2
    
    def test_parse_meddra_missing_directory(self):
        """Test error handling for missing MedDRA directory."""
        missing_dir = Path("nonexistent_meddra")
        with pytest.raises(FileNotFoundError):
            parse_meddra_annotations(missing_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

