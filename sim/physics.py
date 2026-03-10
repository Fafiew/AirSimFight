"""
Physics world wrapper around PyBullet.
"""

import numpy as np
import pybullet as p
import pybullet_data
import logging
from typing import Optional, Sequence, List, Dict, Any

logger = logging.getLogger(__name__)


class PhysicsWorld:
    """
    Wraps PyBullet for physics simulation.
    Supports both headless (DIRECT) and GUI modes.
    """
    
    def __init__(self, config: dict, headless: bool = True):
        """
        Initialize physics world.
        
        Args:
            config: Configuration dictionary with physics parameters
            headless: If True, use DIRECT mode (no GUI). If False, use GUI.
        """
        self.config = config
        self.headless = headless
        self._body_id_map: Dict[str, int] = {}
        self._id_to_body: Dict[int, str] = {}
        
        # Initialize PyBullet
        if headless:
            self.client_id = p.connect(p.DIRECT)
        else:
            self.client_id = p.connect(p.GUI)
        
        # Load standard data (for plane, etc.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set gravity
        gravity = config.get('physics', {}).get('gravity', -9.81)
        p.setGravity(0, 0, gravity)
        
        # Set time step
        self.timestep = config.get('sim', {}).get('timestep', 0.02)
        
        logger.info(f"PhysicsWorld initialized (headless={headless}, gravity={gravity})")
    
    def step(self, dt: float) -> None:
        """
        Step the simulation forward.
        
        Args:
            dt: Time step in seconds
        """
        # Calculate number of substeps
        timestep_substep = self.config.get('sim', {}).get('timestep_substep', 0.005)
        substeps = max(1, int(dt / timestep_substep))
        
        for _ in range(substeps):
            p.stepSimulation()
    
    def add_entity(self, entity_id: str, mass: float, position: np.ndarray, 
                   collision_shape: int = p.GEOM_SPHERE, radius: float = 1.0,
                   orientation: Optional[np.ndarray] = None) -> int:
        """
        Add an entity to the physics world.
        
        Args:
            entity_id: Unique identifier for the entity
            mass: Mass in kg
            position: 3D position [x, y, z]
            collision_shape: PyBullet collision shape type
            radius: Radius for sphere collision shape
            orientation: Quaternion [x, y, z, w]
            
        Returns:
            PyBullet body ID
        """
        if orientation is None:
            orientation = [0, 0, 0, 1]
        
        # Create collision shape
        collision_id = p.createCollisionShape(collision_shape, radius=radius)
        
        # Create multi-body
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            basePosition=position.tolist() if isinstance(position, np.ndarray) else position,
            baseOrientation=orientation
        )
        
        # Store mapping
        self._body_id_map[entity_id] = body_id
        self._id_to_body[body_id] = entity_id
        
        logger.debug(f"Added entity {entity_id} with body_id={body_id}")
        return body_id
    
    def remove_entity(self, entity_id: str) -> None:
        """
        Remove an entity from the physics world.
        
        Args:
            entity_id: Entity identifier to remove
        """
        if entity_id in self._body_id_map:
            body_id = self._body_id_map[entity_id]
            p.removeBody(body_id)
            del self._body_id_map[entity_id]
            del self._id_to_body[body_id]
            logger.debug(f"Removed entity {entity_id}")
    
    def apply_force(self, entity_body_id: int, force: Sequence[float], 
                    pos: Optional[Sequence[float]] = None) -> None:
        """
        Apply a force to an entity.
        
        Args:
            entity_body_id: PyBullet body ID
            force: Force vector [fx, fy, fz]
            pos: Position where to apply force (center of mass if None)
        """
        if pos is None:
            pos = [0, 0, 0]
        p.applyExternalForce(
            entity_body_id,
            -1,  # Link index -1 means base
            force,
            pos,
            p.LINK_FRAME
        )
    
    def get_velocity(self, entity_body_id: int) -> np.ndarray:
        """
        Get the linear velocity of an entity.
        
        Args:
            entity_body_id: PyBullet body ID
            
        Returns:
            Velocity vector [vx, vy, vz]
        """
        vel = p.getBaseVelocity(entity_body_id)
        return np.array(vel[0])
    
    def get_position(self, entity_body_id: int) -> np.ndarray:
        """
        Get the position of an entity.
        
        Args:
            entity_body_id: PyBullet body ID
            
        Returns:
            Position vector [x, y, z]
        """
        pos = p.getBasePositionAndOrientation(entity_body_id)[0]
        return np.array(pos)
    
    def get_orientation(self, entity_body_id: int) -> np.ndarray:
        """
        Get the orientation of an entity as quaternion.
        
        Args:
            entity_body_id: PyBullet body ID
            
        Returns:
            Quaternion [x, y, z, w]
        """
        orient = p.getBasePositionAndOrientation(entity_body_id)[1]
        return np.array(orient)
    
    def set_mass(self, entity_body_id: int, mass: float) -> None:
        """
        Set the mass of an entity.
        
        Args:
            entity_body_id: PyBullet body ID
            mass: New mass in kg
        """
        # PyBullet doesn't allow direct mass change, so we need to rebuild
        # For simplicity, we'll use a different approach - store mass separately
        # and apply forces based on it
        pass
    
    def raycast(self, origin: Sequence[float], direction: Sequence[float], 
                max_dist: float) -> Optional[Dict[str, Any]]:
        """
        Cast a ray and return hit information.
        
        Args:
            origin: Origin point [x, y, z]
            direction: Direction vector [dx, dy, dz] (will be normalized)
            max_dist: Maximum distance to cast
            
        Returns:
            Dict with hit info or None if no hit
        """
        # Normalize direction
        direction = np.array(direction)
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        # Perform raycast
        result = p.rayTest(
            rayFromPosition=origin,
            rayToPosition=(np.array(origin) + direction * max_dist).tolist()
        )
        
        if result[0][0] == -1:
            return None
        
        hit_object_id = result[0][0]
        hit_position = result[0][3]
        hit_fraction = result[0][2]
        
        return {
            'body_id': hit_object_id,
            'position': np.array(hit_position),
            'distance': hit_fraction * max_dist,
            'entity_id': self._id_to_body.get(hit_object_id)
        }
    
    def get_forward_vector(self, entity_body_id: int) -> np.ndarray:
        """
        Get the forward direction vector based on entity orientation.
        
        Args:
            entity_body_id: PyBullet body ID
            
        Returns:
            Forward direction vector
        """
        orient = self.get_orientation(entity_body_id)
        # Convert quaternion to rotation matrix
        return self._quaternion_to_forward(orient)
    
    def _quaternion_to_forward(self, quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to forward direction vector.
        
        Args:
            quat: Quaternion [x, y, z, w]
            
        Returns:
            Forward vector [fx, fy, fz]
        """
        x, y, z, w = quat
        return np.array([
            2 * (x*z + w*y),
            2 * (y*z - w*x),
            1 - 2 * (x*x + y*y)
        ])
    
    def close(self) -> None:
        """Disconnect from PyBullet."""
        p.disconnect(self.client_id)
        logger.info("PhysicsWorld closed")
    
    def reset(self) -> None:
        """Reset the physics world."""
        p.resetSimulation()
        p.setGravity(0, 0, self.config.get('physics', {}).get('gravity', -9.81))
        self._body_id_map.clear()
        self._id_to_body.clear()
