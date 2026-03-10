"""
JSON loader for visualization scenarios.
Validates that JSON files contain only position data (no dynamics).
"""

import json
import jsonschema
from typing import Dict, List, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# JSON Schema for visualization files
VISUALIZATION_SCHEMA = {
    "type": "object",
    "required": ["world", "attackers", "defenders", "targets"],
    "properties": {
        "world": {
            "type": "object",
            "required": ["size"],
            "properties": {
                "size": {"type": "number"}
            }
        },
        "attackers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "position"],
                "properties": {
                    "id": {"type": "string"},
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3
                    }
                },
                "additionalProperties": False  # Strict - no extra fields!
            }
        },
        "defenders": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "position"],
                "properties": {
                    "id": {"type": "string"},
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3
                    }
                },
                "additionalProperties": False  # Strict - no extra fields!
            }
        },
        "targets": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "position"],
                "properties": {
                    "id": {"type": "string"},
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3
                    }
                },
                "additionalProperties": False  # Strict - no extra fields!
            }
        }
    },
    "additionalProperties": False  # Strict - only these keys allowed!
}


def validate_visualization_json(data: Dict) -> bool:
    """
    Validate visualization JSON data against schema.
    
    Raises:
        jsonschema.ValidationError: If JSON doesn't match schema
        
    Args:
        data: Parsed JSON data
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails with details
    """
    try:
        jsonschema.validate(instance=data, schema=VISUALIZATION_SCHEMA)
        return True
    except jsonschema.ValidationError as e:
        raise ValueError(f"Invalid visualization JSON: {e.message}")


def load_positions(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load positions from a visualization JSON file.
    
    This function validates that the JSON is a proper visualization file
    (positions only) and rejects files with dynamics data.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Dict with 'attackers', 'defenders', 'targets' lists
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or contains extra fields
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    
    # Load JSON
    with open(path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}")
    
    # Validate against schema
    try:
        validate_visualization_json(data)
    except jsonschema.ValidationError as e:
        raise ValueError(f"Visualization JSON validation failed: {e.message}")
    
    # Return structured data
    result = {
        'world': data.get('world', {}),
        'attackers': data.get('attackers', []),
        'defenders': data.get('defenders', []),
        'targets': data.get('targets', [])
    }
    
    logger.info(f"Loaded visualization from {path}: "
                f"{len(result['attackers'])} attackers, "
                f"{len(result['defenders'])} defenders, "
                f"{len(result['targets'])} targets")
    
    return result


def load_scenario(path: str) -> Dict[str, Any]:
    """
    Load a scenario and return entity instances.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Dict with entity lists as PositionData named tuples
    """
    from collections import namedtuple
    
    PositionData = namedtuple('PositionData', ['id', 'position'])
    
    data = load_positions(path)
    
    return {
        'world': data['world'],
        'attackers': [PositionData(a['id'], a['position']) for a in data['attackers']],
        'defenders': [PositionData(d['id'], d['position']) for d in data['defenders']],
        'targets': [PositionData(t['id'], t['position']) for t in data['targets']]
    }


# CLI for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python json_loader.py <scenario.json>")
        sys.exit(1)
    
    try:
        data = load_positions(sys.argv[1])
        print("Valid visualization JSON!")
        print(f"  Attackers: {len(data['attackers'])}")
        print(f"  Defenders: {len(data['defenders'])}")
        print(f"  Targets: {len(data['targets'])}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
