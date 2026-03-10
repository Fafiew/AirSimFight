"""
Unit tests for physics.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.physics import PhysicsWorld
from sim.entities import Attacker


class TestPhysics:
    """Tests for physics world."""
    
    @pytest.fixture
    def config(self):
        """Get test configuration."""
        return {
            'sim': {
                'timestep': 0.02,
                'timestep_substep': 0.005,
                'world_size': 20000,
                'boundary_timeout_seconds': 5
            },
            'physics': {
                'gravity': -9.81,
                'drag_coefficient_default': 0.01
            },
            'attacker_dynamics': {
                'mass_empty': 120.0,
                'fuel_capacity': 50.0,
                'fuel_mass_initial': 50.0,
                'engine': {
                    'thrust_max': 8000.0,
                    'throttle_response_time': 0.1,
                    'fuel_consumption_rate': 0.00012
                },
                'aerodynamics': {
                    'drag_coefficient': 0.02,
                    'reference_area': 0.8,
                    'air_density': 1.225
                }
            }
        }
    
    @pytest.fixture
    def physics_world(self, config):
        """Create headless physics world."""
        world = PhysicsWorld(config, headless=True)
        yield world
        world.close()
    
    def test_physics_world_init(self, physics_world):
        """Test physics world initializes correctly."""
        assert physics_world is not None
        assert physics_world.headless is True
    
    def test_add_entity(self, physics_world):
        """Test adding entity to physics world."""
        body_id = physics_world.add_entity(
            entity_id="test_entity",
            mass=10.0,
            position=np.array([0, 0, 10]),
            radius=1.0
        )
        
        assert body_id >= 0
        assert "test_entity" in physics_world._body_id_map
    
    def test_apply_force(self, physics_world):
        """Test applying force to entity."""
        body_id = physics_world.add_entity(
            entity_id="test_force",
            mass=1.0,
            position=np.array([0, 0, 10]),
            radius=1.0
        )
        
        # Apply constant force
        force = [100, 0, 0]
        physics_world.apply_force(body_id, force)
        
        # Step simulation
        physics_world.step(0.02)
        
        # Check velocity changed
        vel = physics_world.get_velocity(body_id)
        assert vel[0] > 0  # Should have some velocity in x direction
    
    def test_get_velocity(self, physics_world):
        """Test getting entity velocity."""
        body_id = physics_world.add_entity(
            entity_id="test_vel",
            mass=1.0,
            position=np.array([0, 0, 10]),
            radius=1.0
        )
        
        vel = physics_world.get_velocity(body_id)
        assert isinstance(vel, np.ndarray)
        assert vel.shape == (3,)
    
    def test_raycast(self, physics_world):
        """Test raycasting."""
        # Add two entities
        body1 = physics_world.add_entity(
            entity_id="ray_origin",
            mass=1.0,
            position=np.array([0, 0, 10]),
            radius=1.0
        )
        body2 = physics_world.add_entity(
            entity_id="ray_target",
            mass=1.0,
            position=np.array([100, 0, 10]),
            radius=1.0
        )
        
        # Raycast from body1 toward body2
        result = physics_world.raycast(
            origin=[0, 0, 10],
            direction=[1, 0, 0],
            max_dist=200
        )
        
        assert result is not None
        assert 'distance' in result
    
    def test_attacker_force_integration(self, config, physics_world):
        """Test that attacker responds to thrust force."""
        # Create attacker
        attacker = Attacker(
            entity_id="test_attacker",
            position=np.array([0, 0, 1000]),
            config=config
        )
        
        # Add to physics
        body_id = physics_world.add_entity(
            attacker.id,
            mass=attacker.mass_current,
            position=attacker.position,
            radius=2.0
        )
        attacker.body_id = body_id
        
        # Get initial velocity
        initial_vel = physics_world.get_velocity(body_id)
        
        # Apply thrust action (full throttle)
        action = {
            'throttle': 1.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'roll': 0.0
        }
        
        # Step multiple times
        dt = config['sim']['timestep']
        for _ in range(10):
            attacker.step(action, dt, physics_world)
            physics_world.step(dt)
        
        # Get final velocity
        final_vel = physics_world.get_velocity(body_id)
        
        # Velocity should have increased (thrust > drag)
        speed_increase = np.linalg.norm(final_vel) - np.linalg.norm(initial_vel)
        assert speed_increase > 0  # Should have accelerated
    
    def test_mass_updates_with_fuel(self, config):
        """Test that mass updates when fuel is consumed."""
        attacker = Attacker(
            entity_id="test_mass",
            position=np.array([0, 0, 1000]),
            config=config
        )
        
        initial_mass = attacker.mass_current
        
        # Apply throttle to consume fuel
        action = {'throttle': 1.0, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
        for _ in range(100):
            attacker.step(action, 0.02, None)  # No physics world needed for fuel consumption
        
        final_mass = attacker.mass_current
        
        # Mass should decrease as fuel is consumed
        assert final_mass < initial_mass
        assert attacker.fuel < config['attacker_dynamics']['fuel_mass_initial']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
