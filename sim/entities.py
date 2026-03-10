"""
Entity classes for AirSimFight simulation.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class Entity:
    """
    Base class for all simulation entities.
    """
    
    def __init__(self, entity_id: str, position: np.ndarray):
        """
        Initialize entity.
        
        Args:
            entity_id: Unique identifier
            position: 3D position [x, y, z]
        """
        self.id = entity_id
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.body_id: Optional[int] = None
        self.orientation = np.array([0, 0, 0, 1], dtype=np.float32)  # quaternion
    
    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, pos={self.position})"


class Attacker(Entity):
    """
    Attacker aircraft with thrust/fuel dynamics.
    Speed is physics-driven by thrust and fuel - no artificial speed limit.
    """
    
    def __init__(
        self,
        entity_id: str,
        position: np.ndarray,
        config: dict,
        can_target_defenders: bool = False,
        is_pretrained: bool = False
    ):
        super().__init__(entity_id, position)
        
        # Dynamics parameters from config
        dynamics = config.get('attacker_dynamics', {})
        
        self.mass_empty = dynamics.get('mass_empty', 120.0)
        self.fuel_capacity = dynamics.get('fuel_capacity', 50.0)
        self.fuel = dynamics.get('fuel_mass_initial', 50.0)
        
        # Engine parameters
        engine = dynamics.get('engine', {})
        self.engine_thrust_max = engine.get('thrust_max', 8000.0)
        self.throttle_response_time = engine.get('throttle_response_time', 0.1)
        self.fuel_consumption_rate = engine.get('fuel_consumption_rate', 0.00012)
        
        # Aerodynamics
        aero = dynamics.get('aerodynamics', {})
        self.drag_coefficient = aero.get('drag_coefficient', 0.02)
        self.reference_area = aero.get('reference_area', 0.8)
        self.air_density = aero.get('air_density', 1.225)
        
        # State
        self.current_throttle = 0.0
        self.health = 100.0
        self.is_alive = True
        self.can_target_defenders = can_target_defenders
        self.is_pretrained = is_pretrained
        
        # Computed mass (empty + fuel)
        self.mass = self.mass_empty + self.fuel
        
        # Current thrust (computed each step)
        self.current_thrust = 0.0
        
        logger.debug(f"Attacker {self.id}: mass={self.mass}, fuel={self.fuel}, thrust_max={self.engine_thrust_max}")
    
    @property
    def mass_current(self) -> float:
        """Current mass (empty + remaining fuel)."""
        return self.mass_empty + self.fuel
    
    def step(self, action: Dict, dt: float, physics_world) -> None:
        """
        Update attacker dynamics for one time step.
        
        Args:
            action: Dict with 'throttle' (0-1), 'pitch', 'yaw', 'roll' (radians or rates)
            dt: Time step in seconds
            physics_world: PhysicsWorld instance
        """
        if not self.is_alive:
            return
        
        # Extract action components
        throttle = float(action.get('throttle', 0.0))
        throttle = np.clip(throttle, 0.0, 1.0)
        
        # Compute thrust based on fuel
        if self.fuel > 0:
            self.current_throttle = throttle
            self.current_thrust = throttle * self.engine_thrust_max
        else:
            # No fuel = no thrust
            self.current_throttle = 0.0
            self.current_thrust = 0.0
        
        # Consume fuel
        fuel_used = self.current_thrust * self.fuel_consumption_rate * dt
        fuel_used = min(fuel_used, self.fuel)  # Can't consume more than available
        self.fuel -= fuel_used
        self.fuel = max(0.0, self.fuel)
        
        # Get forward direction from orientation
        forward = physics_world.get_forward_vector(self.body_id) if self.body_id is not None else np.array([1, 0, 0])
        
        # Apply thrust force
        thrust_force = forward * self.current_thrust
        
        # Compute drag (quadratic with velocity)
        velocity = physics_world.get_velocity(self.body_id) if self.body_id is not None else np.zeros(3)
        speed = np.linalg.norm(velocity)
        
        if speed > 0:
            drag_magnitude = 0.5 * self.air_density * self.drag_coefficient * self.reference_area * speed ** 2
            drag_force = -velocity / speed * drag_magnitude
        else:
            drag_force = np.zeros(3)
        
        # Total force - apply multiple times per step for better acceleration
        total_force = thrust_force + drag_force
        
        # Apply forces via physics - scale thrust significantly for visible aircraft acceleration
        # (PyBullet needs higher forces for realistic SI unit physics)
        if self.body_id is not None and self.current_thrust > 0:
            # Scale force significantly for aircraft-like acceleration
            effective_force = total_force * 100.0  # Much larger scale for realistic acceleration
            physics_world.apply_force(self.body_id, effective_force.tolist())
        
        # Update health if fuel depleted (structural stress simulation)
        if self.fuel == 0 and self.current_thrust > 0:
            # No thrust possible - consider this "disabled"
            pass
    
    def apply_damage(self, amount: float) -> None:
        """
        Apply damage to attacker.
        
        Args:
            amount: Damage amount to subtract from health
        """
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
            logger.info(f"Attacker {self.id} destroyed")
    
    def choose_action(self, observation: Dict) -> np.ndarray:
        """
        Wrapper to call RL policy (implemented in rl/policies.py).
        This is a placeholder that should be replaced with actual policy call.
        
        Args:
            observation: Current observation dict
            
        Returns:
            Action array [throttle, pitch, yaw, roll]
        """
        # Placeholder - actual implementation in rl/policies.py
        return np.array([0.5, 0.0, 0.0, 0.0])
    
    def get_observation(self) -> Dict[str, Any]:
        """
        Get current observation state.
        
        Returns:
            Dict with position, velocity, fuel, health, etc.
        """
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'fuel': self.fuel,
            'fuel_capacity': self.fuel_capacity,
            'mass': self.mass_current,
            'health': self.health,
            'is_alive': self.is_alive,
            'current_throttle': self.current_throttle,
            'current_thrust': self.current_thrust,
        }


class Defender(Entity):
    """
    Defender with radar and interceptor launching capability.
    """
    
    def __init__(self, entity_id: str, position: np.ndarray, config: dict):
        super().__init__(entity_id, position)
        
        # Defender parameters
        defaults = config.get('defender_defaults', {})
        self.radar_range = defaults.get('radar_range', 15000.0)
        self.interceptor_speed = defaults.get('intercept', 600.0)
        self.cooldown = defaults.get('cooldown', 2.0)
        self.health = defaults.get('health', 100.0)
        self.is_alive = True
        
        # Radar config
        radar = defaults.get('radar', {})
        self.radar_h_fov = np.deg2rad(radar.get('horizontal_fov_deg', 120.0))
        self.radar_v_fov = np.deg2rad(radar.get('vertical_fov_deg', 30.0))
        
        # State
        self.last_launch_time = -1000.0  # Large negative to allow immediate launch
        self.interceptors: List['Interceptor'] = []
        
        # Sensor - will be set by environment
        self.radar_sensor = None
    
    def scan(self, environment) -> List[Dict]:
        """
        Scan for targets using radar.
        
        Args:
            environment: The AirCombatEnv instance
            
        Returns:
            List of detection dicts {id, bearing, range, rcs, confidence}
        """
        detections = []
        
        if not self.is_alive or self.radar_sensor is None:
            return detections
        
        # Get all attackers
        attackers = getattr(environment, 'attackers', {})
        
        for attacker_id, attacker in attackers.items():
            if not attacker.is_alive:
                continue
            
            # Use radar sensor to scan
            detections = self.radar_sensor.scan(self, environment.physics_world, list(environment.entities.values()))
            break  # Only need to scan once
        
        return detections
    
    def launch_interceptor(self, target_id: str, physics_world) -> 'Interceptor':
        """
        Launch an interceptor at a target.
        
        Args:
            target_id: ID of target attacker
            physics_world: PhysicsWorld instance
            
        Returns:
            New Interceptor entity
        """
        import time
        current_time = time.time()
        
        if current_time - self.last_launch_time < self.cooldown:
            logger.warning(f"Defender {self.id} on cooldown")
            return None
        
        if not self.is_alive:
            return None
        
        # Create interceptor
        interceptor_id = f"{self.id}_int_{len(self.interceptors)}"
        interceptor = Interceptor(
            entity_id=interceptor_id,
            position=self.position.copy(),
            speed=self.interceptor_speed,
            target_id=target_id,
            config=physics_world.config
        )
        
        # Add to physics
        body_id = physics_world.add_entity(
            interceptor_id,
            mass=10.0,
            position=interceptor.position,
            radius=0.5
        )
        interceptor.body_id = body_id
        
        self.last_launch_time = current_time
        self.interceptors.append(interceptor)
        
        logger.info(f"Defender {self.id} launched interceptor {interceptor_id} at {target_id}")
        return interceptor
    
    def apply_damage(self, amount: float) -> None:
        """
        Apply damage to defender.
        
        Args:
            amount: Damage amount
        """
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
            logger.info(f"Defender {self.id} destroyed")


class Interceptor(Entity):
    """
    Interceptor missile/projectile with guidance.
    """
    
    def __init__(
        self,
        entity_id: str,
        position: np.ndarray,
        speed: float,
        target_id: str,
        config: dict
    ):
        super().__init__(entity_id, position)
        
        self.speed = speed
        self.target_id = target_id
        self.guidance_mode = 'proportional_navigation'
        self.is_active = True
        self.guidance_gain = 3.0
        
        # Initial velocity toward target direction
        self.velocity = np.array([speed, 0, 0], dtype=np.float32)
        
        logger.debug(f"Interceptor {self.id} created targeting {target_id}")
    
    def update(self, dt: float, physics_world, target_position: np.ndarray) -> None:
        """
        Update interceptor position with guidance.
        
        Args:
            dt: Time step
            physics_world: PhysicsWorld instance
            target_position: Current position of target
        """
        if not self.is_active:
            return
        
        # Compute direction to target
        to_target = target_position - self.position
        distance = np.linalg.norm(to_target)
        
        if distance < 1.0:
            # Hit target
            self.is_active = False
            logger.info(f"Interceptor {self.id} reached target")
            return
        
        # Normalize direction
        target_direction = to_target / distance
        
        # Current velocity direction
        current_direction = self.velocity / (np.linalg.norm(self.velocity) + 1e-8)
        
        # Proportional navigation guidance
        relative_velocity = np.zeros(3)  # Simplified - assume target stationary
        closing_velocity = -np.dot(relative_velocity, target_direction)
        
        # Lateral acceleration for navigation
        accel = self.guidance_gain * closing_velocity * np.cross(current_direction, target_direction)
        
        # Update velocity
        self.velocity = self.velocity + accel * dt
        
        # Clamp to max speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.speed:
            self.velocity = self.velocity / speed * self.speed
        
        # Update position
        self.position = self.position + self.velocity * dt
        
        # Update physics body
        if self.body_id is not None:
            physics_world.apply_force(self.body_id, (self.velocity * 0.1).tolist())
            self.position = physics_world.get_position(self.body_id)
            self.velocity = physics_world.get_velocity(self.body_id)


class Target(Entity):
    """
    Target object for attackers to reach.
    """
    
    def __init__(self, entity_id: str, position: np.ndarray, value: float = 100.0):
        super().__init__(entity_id, position)
        self.value = value
        self.is_captured = False
    
    def __repr__(self):
        return f"Target(id={self.id}, pos={self.position}, value={self.value})"
