"""
Panda3D renderer for AirSimFight visualization.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Try to import Panda3D
PANDAS_AVAILABLE = False
ShowBase = None

try:
    import panda3d.core as pc
    PANDAS_AVAILABLE = True
    
    # Try to import ShowBase for rendering
    try:
        from direct.showbase.ShowBase import ShowBase as _ShowBase
        ShowBase = _ShowBase
    except ImportError:
        pass
        
except ImportError:
    pass


class AirCombatRenderer:
    """
    Panda3D renderer for air combat visualization.
    """
    
    def __init__(
        self,
        env,
        window_title: str = "AirSimFight",
        width: int = 1280,
        height: int = 720,
        show_sensor_cones: bool = True,
        show_hud: bool = True
    ):
        """
        Initialize renderer.
        
        Args:
            env: AirCombatEnv instance
            window_title: Window title
            width: Window width
            height: Window height
            show_sensor_cones: Whether to show sensor FOV cones
            show_hud: Whether to show HUD elements
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Panda3D not available. Install with: pip install panda3d")
        
        self.env = env
        self.width = width
        self.height = height
        self.show_sensor_cones = show_sensor_cones
        self.show_hud = show_hud
        
        # State
        self.paused = False
        self.entity_nodes = {}
        self.sensor_cones = {}
        
        logger.info("Renderer initialized (Panda3D available)")
    
    def update(self) -> None:
        """Update visualization for current frame."""
        if self.paused:
            return
        
        # Update all entity positions
        for attacker_id, attacker in self.env.attackers.items():
            if attacker.is_alive:
                self._update_entity(attacker_id, attacker.position)
        
        for defender_id, defender in self.env.defenders.items():
            if defender.is_alive:
                self._update_entity(defender_id, defender.position)
    
    def _update_entity(self, entity_id: str, position: np.ndarray) -> None:
        """Update entity position."""
        # Visualization would be updated here if Panda3D scene is running
        pass
    
    def run(self) -> None:
        """Start the rendering loop."""
        logger.info("Renderer run() called - full visualization requires ShowBase")


def create_renderer(env, headless: bool = False, **kwargs):
    """
    Create a Renderer instance.
    
    Args:
        env: AirCombatEnv instance
        headless: If True, return None (for headless mode)
        **kwargs: Additional args for renderer
        
    Returns:
        Renderer instance or None
    """
    if headless:
        return None
    
    if not PANDAS_AVAILABLE:
        print("Panda3D not found. Please install Panda3D or run in headless mode.")
        print("Install with: pip install panda3d")
        return None
    
    try:
        return AirCombatRenderer(env, **kwargs)
    except Exception as e:
        logger.error(f"Error creating renderer: {e}")
        return None


def check_panda3d() -> bool:
    """
    Check if Panda3D is available.
    
    Returns:
        True if Panda3D is installed
    """
    return PANDAS_AVAILABLE
