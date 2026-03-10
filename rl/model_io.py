"""
Model I/O with atomic save/load functionality.
"""

import os
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch

logger = logging.getLogger(__name__)


def atomic_save(data: Any, path: str) -> None:
    """
    Save data atomically by writing to temp file then renaming.
    
    Args:
        data: Data to save
        path: Target path
    """
    path_obj = Path(path)
    parent_dir = path_obj.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file in same directory
    fd, temp_path = tempfile.mkstemp(dir=parent_dir, suffix='.tmp')
    
    try:
        with os.fdopen(fd, 'wb') as f:
            if isinstance(data, torch.nn.Module):
                torch.save(data.state_dict(), f)
            else:
                torch.save(data, f)
        
        # Atomic rename
        os.replace(temp_path, path)
        logger.debug(f"Atomically saved to {path}")
        
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def atomic_save_dict(state_dict: Dict, path: str) -> None:
    """
    Save state dict atomically.
    
    Args:
        state_dict: State dict to save
        path: Target path
    """
    path_obj = Path(path)
    parent_dir = path_obj.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    fd, temp_path = tempfile.mkstemp(dir=parent_dir, suffix='.tmp')
    
    try:
        with os.fdopen(fd, 'wb') as f:
            torch.save(state_dict, f)
        
        os.replace(temp_path, path)
        logger.debug(f"Atomically saved state dict to {path}")
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def save_model(model: torch.nn.Module, path: str, format: str = 'pth') -> None:
    """
    Save model with atomic write.
    
    Args:
        model: PyTorch model
        path: Save path (without extension)
        format: 'pth', 'zip', or 'both'
    """
    path = str(path)  # Ensure string
    
    if format in ['pth', 'both']:
        pth_path = f"{path}.pth"
        atomic_save(model, pth_path)
        logger.info(f"Model saved to {pth_path}")
    
    # Note: SB3 models use their own save format, handled separately
    if format == 'zip':
        # This would be handled by SB3's .save() method
        pass


def load_model(path: str, device: torch.device = None) -> Optional[Dict]:
    """
    Load model weights from file.
    
    Args:
        path: Path to model file (.pth or .zip)
        device: Device to load to
        
    Returns:
        State dict or None if not found
    """
    if not os.path.exists(path):
        logger.warning(f"Model file not found: {path}")
        return None
    
    if path.endswith('.pth'):
        try:
            state_dict = torch.load(path, map_location=device)
            logger.info(f"Loaded model from {path}")
            return state_dict
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
            return None
    
    # For .zip files, return path for SB3 to handle
    return path


def load_sb3_model(path: str, device: str = 'cpu'):
    """
    Load Stable-Baselines3 model.
    
    Args:
        path: Path to .zip file
        device: Device to load on
        
    Returns:
        Loaded SB3 model
    """
    try:
        from stable_baselines3 import PPO
        model = PPO.load(path, device=device)
        logger.info(f"Loaded SB3 model from {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading SB3 model: {e}")
        return None


def save_sb3_model(model, path: str) -> None:
    """
    Save Stable-Baselines3 model atomically.
    
    Args:
        model: SB3 model
        path: Save path (will add .zip)
    """
    if not path.endswith('.zip'):
        path = f"{path}.zip"
    
    # SB3's save writes directly, so we wrap it
    path_obj = Path(path)
    parent_dir = path_obj.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    fd, temp_path = tempfile.mkstemp(dir=parent_dir, suffix='.zip')
    os.close(fd)
    os.remove(temp_path)  # SB3 will create this
    
    try:
        model.save(path.replace('.zip', ''))  # SB3 adds .zip automatically
        logger.info(f"Saved SB3 model to {path}")
    except Exception as e:
        logger.error(f"Error saving SB3 model: {e}")
        raise


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the latest checkpoint file in a directory.
    
    Args:
        checkpoint_dir: Directory to search
        
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pth"))
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def list_checkpoints(checkpoint_dir: str) -> list:
    """
    List all checkpoint files in a directory.
    
    Args:
        checkpoint_dir: Directory to search
        
    Returns:
        List of checkpoint paths sorted by step number
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pth"))
    
    # Extract step numbers and sort
    def get_step(p):
        try:
            return int(p.stem.split('_')[1])
        except:
            return 0
    
    return sorted(checkpoints, key=get_step)
