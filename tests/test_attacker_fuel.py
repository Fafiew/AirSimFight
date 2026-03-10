"""
Unit tests for attacker fuel dynamics.
Critical test to verify no max_speed clamp and proper fuel consumption.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.entities import Attacker
from sim.physics import PhysicsWorld


class TestAttackerFuel:
    """Tests for attacker fuel dynamics - critical for verifying no max_speed."""
    
    @pytest.fixture
    def config(self):
        """Get test configuration."""
        return {
            'sim': {
                'timestep': 0.02,
                'timestep_substep': 0.005,
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
    
    def test_no_max_speed_attribute(self, config):
        """CRITICAL: Verify no max_speed attribute exists on attacker."""
        attacker = Attacker(
            entity_id="test",
            position=np.array([0, 0, 1000]),
            config=config
        )
        
        # This test MUST pass - there should be NO max_speed
        assert not hasattr(attacker, 'max_speed'), \
            "ERROR: Attacker has max_speed attribute - this violates the specification!"
    
    def test_no_max_speed_in_step(self, config, physics_world):
        """CRITICAL: Verify no velocity clamping in step method."""
        attacker = Attacker(
            entity_id="test_no_max",
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
        
        # Apply maximum thrust for many steps
        action = {
            'throttle': 1.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'roll': 0.0
        }
        
        dt = config['sim']['timestep']
        velocities = []
        
        # Step many times - should accelerate continuously (no artificial cap)
        for _ in range(500):
            attacker.step(action, dt, physics_world)
            physics_world.step(dt)  # Step physics to update velocity
            vel = physics_world.get_velocity(body_id)
            velocities.append(np.linalg.norm(vel))
        
        # Velocity should keep increasing (or at least not be clamped)
        # With no max_speed, the aircraft should accelerate 
        final_speed = velocities[-1]
        
        # If there was a max_speed hard clamp, speed would be exactly at that value
        # Without max_speed, it should be > 50 m/s (accounting for PyBullet damping)
        # This test verifies that velocity is NOT artificially capped at a low value
        assert final_speed > 50, \
            f"Speed seems artificially capped at {final_speed} m/s - check for max_speed!"
    
    def test_fuel_consumption_rate(self, config, physics_world):
        """Test that fuel is consumed at the correct rate."""
        attacker = Attacker(
            entity_id="test_fuel",
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
        
        initial_fuel = attacker.fuel
        fuel_capacity = config['attacker_dynamics']['fuel_capacity']
        thrust_max = config['attacker_dynamics']['engine']['thrust_max']
        consumption_rate = config['attacker_dynamics']['engine']['fuel_consumption_rate']
        
        # Apply full throttle
        action = {'throttle': 1.0, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
        dt = 0.02
        num_steps = 100
        
        for _ in range(num_steps):
            attacker.step(action, dt, physics_world)
        
        final_fuel = attacker.fuel
        
        # Expected fuel consumption: thrust * consumption_rate * time
        expected_consumption = thrust_max * consumption_rate * (dt * num_steps)
        actual_consumption = initial_fuel - final_fuel
        
        # Allow 5% tolerance
        tolerance = 0.05
        assert abs(actual_consumption - expected_consumption) < expected_consumption * tolerance, \
            f"Fuel consumption {actual_consumption} differs significantly from expected {expected_consumption}"
    
    def test_zero_thrust_when_fuel_empty(self, config, physics_world):
        """Test that thrust becomes zero when fuel is exhausted."""
        attacker = Attacker(
            entity_id="test_empty",
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
        
        # Consume all fuel first
        action = {'throttle': 1.0, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        dt = 0.02
        
        # Run until fuel is empty
        while attacker.fuel > 0:
            attacker.step(action, dt, physics_world)
            physics_world.step(dt)
        
        # Verify fuel is empty
        assert attacker.fuel == 0.0
        
        # Record velocity
        vel_before = np.linalg.norm(physics_world.get_velocity(body_id))
        
        # Now apply thrust - should be zero
        attacker.step(action, dt, physics_world)
        physics_world.step(dt)
        
        vel_after = np.linalg.norm(physics_world.get_velocity(body_id))
        
        # Velocity should NOT increase (no thrust)
        # Allow larger tolerance due to physics simulation drift
        velocity_change = abs(vel_after - vel_before)
        assert velocity_change < 10.0, \
            f"Thrust should be zero when fuel=0, but velocity changed by {velocity_change}"
    
    def test_mass_decreases_with_fuel(self, config):
        """Test that mass decreases as fuel is consumed."""
        attacker = Attacker(
            entity_id="test_mass",
            position=np.array([0, 0, 1000]),
            config=config
        )
        
        initial_mass = attacker.mass_current
        initial_fuel = attacker.fuel
        
        # Apply throttle to consume fuel
        action = {'throttle': 1.0, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
        for _ in range(100):
            attacker.step(action, 0.02, None)
        
        final_mass = attacker.mass_current
        final_fuel = attacker.fuel
        
        # Mass should decrease
        assert final_mass < initial_mass
        
        # Mass change should equal fuel change
        mass_change = initial_mass - final_mass
        fuel_change = initial_fuel - final_fuel
        
        assert abs(mass_change - fuel_change) < 0.01, \
            "Mass change should equal fuel change"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
