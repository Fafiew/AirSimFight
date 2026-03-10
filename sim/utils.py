"""
Utility functions for simulation.
"""

import numpy as np
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def load_config(config_path: str, overrides: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML config file
        overrides: Optional dict to override config values
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if overrides:
        config = merge_config(config, overrides)
    
    return config


def merge_config(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge override dict into base config.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    
    return result


def parse_overrides(overrides_str: str) -> Dict:
    """
    Parse JSON string into overrides dict.
    
    Args:
        overrides_str: JSON string with config overrides
        
    Returns:
        Parsed overrides dict
    """
    if not overrides_str:
        return {}
    
    try:
        return json.loads(overrides_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid overrides JSON: {e}")


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two points.
    
    Args:
        a: First point
        b: Second point
        
    Returns:
        Distance
    """
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute angle between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Angle in radians
    """
    a = np.array(a)
    b = np.array(b)
    
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    
    return float(np.arccos(np.clip(dot / (norm_a * norm_b), -1, 1)))


def heading_to_target(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Get heading vector from position to target.
    
    Args:
        position: Current position
        target: Target position
        
    Returns:
        Normalized heading vector
    """
    direction = target - position
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.zeros(3)
    return direction / norm


def sample_in_ring(inner_radius: float, outer_radius: float, height_min: float, height_max: float) -> np.ndarray:
    """
    Sample a random position in a cylindrical ring.
    
    Args:
        inner_radius: Inner radius of ring
        outer_radius: Outer radius of ring
        height_min: Minimum height
        height_max: Maximum height
        
    Returns:
        3D position [x, y, z]
    """
    # Sample radius uniformly in area (r^2 distribution)
    r = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2))
    
    # Sample angle uniformly
    theta = np.random.uniform(0, 2 * np.pi)
    
    # Compute x, y
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Sample height
    z = np.random.uniform(height_min, height_max)
    
    return np.array([x, y, z], dtype=np.float32)


def setup_logging(log_dir: str = "logs", tensorboard: bool = True) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        tensorboard: Whether TensorBoard logging is enabled
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Basic logging config
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{log_dir}/simulation.log")
        ]
    )
    
    logger.info("Logging configured")


def clip_angle(angle: float) -> float:
    """
    Clip angle to [-pi, pi].
    
    Args:
        angle: Angle in radians
        
    Returns:
        Clipped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [0, 2*pi].
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle
    """
    while angle >= 2 * np.pi:
        angle -= 2 * np.pi
    while angle < 0:
        angle += 2 * np.pi
    return angle
