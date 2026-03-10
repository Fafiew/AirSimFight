"""
Unit tests for JSON loader.
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.json_loader import load_positions, validate_visualization_json, VISUALIZATION_SCHEMA


class TestJSONLoader:
    """Tests for JSON visualization loader."""
    
    def test_load_valid_visualization(self):
        """Test loading a valid visualization JSON."""
        # Use the example file
        path = Path(__file__).parent.parent / "scenarios" / "example_visualization.json"
        
        if not path.exists():
            pytest.skip("Example scenario file not found")
        
        data = load_positions(str(path))
        
        assert 'attackers' in data
        assert 'defenders' in data
        assert 'targets' in data
        assert len(data['attackers']) > 0
    
    def test_reject_extra_fields(self):
        """Test that loader rejects JSON with extra entity fields."""
        invalid_json = {
            "world": {"size": 20000},
            "attackers": [
                {
                    "id": "a1",
                    "position": [1000, 0, 500],
                    "velocity": [100, 0, 0],  # Extra field!
                    "health": 100  # Extra field!
                }
            ],
            "defenders": [],
            "targets": []
        }
        
        with pytest.raises(ValueError, match="Invalid visualization JSON"):
            validate_visualization_json(invalid_json)
    
    def test_valid_minimal_json(self):
        """Test that minimal valid JSON passes validation."""
        valid_json = {
            "world": {"size": 20000},
            "attackers": [{"id": "a1", "position": [100, 200, 300]}],
            "defenders": [{"id": "d1", "position": [0, 0, 0]}],
            "targets": [{"id": "t1", "position": [50, 50, 0]}]
        }
        
        # Should not raise
        assert validate_visualization_json(valid_json) is True
    
    def test_missing_required_keys(self):
        """Test that missing required keys are caught."""
        invalid_json = {
            "world": {"size": 20000},
            "attackers": [],
            # Missing "defenders" and "targets"
        }
        
        with pytest.raises(ValueError):
            validate_visualization_json(invalid_json)
    
    def test_invalid_position_format(self):
        """Test that invalid position format is caught."""
        invalid_json = {
            "world": {"size": 20000},
            "attackers": [{"id": "a1", "position": "invalid"}],
            "defenders": [],
            "targets": []
        }
        
        with pytest.raises(ValueError):
            validate_visualization_json(invalid_json)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
