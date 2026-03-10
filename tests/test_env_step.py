"""
Unit tests for environment step.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.environment import AirCombatEnv


class TestEnvironmentStep:
    """Tests for AirCombatEnv step function."""
    
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
                    'throttle_max': 8000.0,
                    'throttle_response_time': 0.1,
                    'fuel_consumption_rate': 0.00012
                },
                'aerodynamics': {
                    'drag_coefficient': 0.02,
                    'reference_area': 0.8,
                    'air_density': 1.225
                }
            },
            'defender_defaults': {
                'radar_range': 15000.0,
                'interceptor_speed': 600.0,
                'cooldown': 2.0,
                'health': 100.0,
                'radar': {
                    'horizontal_fov_deg': 120.0,
                    'vertical_fov_deg': 30.0
                }
            },
            'sensors': {
                'radar': {
                    'default_range': 15000.0,
                    'horizontal_fov_deg': 120.0,
                    'vertical_fov_deg': 30.0,
                    'rcs_noise_std': 0.1
                },
                'ir': {
                    'default_range': 5000.0,
                    'lock_angle_deg': 10.0,
                    'noise_std': 0.05
                }
            },
            'rl': {
                'attacker_rewards': {
                    'target_reward': 100.0,
                    'destroyed_penalty': -100.0,
                    'fuel_cost_coeff': 50.0
                }
            },
            'procedural_spawn': {
                'R_def_min': 50.0,
                'R_def_max': 500.0,
                'R_att_min': 5000.0,
                'R_att_max': 20000.0,
                'H_att_min': 1500.0,
                'H_att_max': 5000.0
            }
        }
    
    @pytest.fixture
    def env(self, config):
        """Create headless environment."""
        env = AirCombatEnv(
            config=config,
            device='cpu',
            headless=True,
            pretrain_mode=True
        )
        yield env
        env.close()
    
    def test_env_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()
        
        assert isinstance(obs, dict)
        assert 'a1' in obs
    
    def test_env_step_returns(self, env):
        """Test that step returns proper types."""
        env.reset()
        
        # Create valid action
        action = {
            'a1': {
                'throttle': 0.5,
                'pitch': 0.0,
                'yaw': 0.0,
                'roll': 0.0
            }
        }
        
        obs, rewards, dones, truncates, infos = env.step(action)
        
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(dones, dict)
        assert isinstance(truncates, dict)
        assert isinstance(infos, dict)
    
    def test_env_step_updates_state(self, env):
        """Test that step updates environment state."""
        env.reset()
        
        # Get initial state
        initial_pos = env.attackers['a1'].position.copy()
        
        # Step
        action = {'a1': {'throttle': 1.0, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}}
        
        for _ in range(10):
            env.step(action)
        
        # Check position changed
        final_pos = env.attackers['a1'].position
        
        # Position should have changed (aircraft should move)
        assert not np.allclose(initial_pos, final_pos)
    
    def test_observation_structure(self, env):
        """Test observation structure."""
        env.reset()
        
        obs, *_ = env.step({'a1': {'throttle': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}})
        
        attacker_obs = obs['a1']
        
        assert 'position' in attacker_obs
        assert 'velocity' in attacker_obs
        assert 'fuel' in attacker_obs
        assert 'health' in attacker_obs
        assert 'radar_detections' in attacker_obs
        assert 'ir_signal' in attacker_obs
    
    def test_dual_mode_spawns_multiple(self, config):
        """Test that dual mode spawns multiple entities."""
        env = AirCombatEnv(
            config=config,
            device='cpu',
            headless=True,
            pretrain_mode=False
        )
        
        env.reset()
        
        # Should have multiple attackers and defenders
        # (exact counts are random 1-10 each)
        assert len(env.attackers) >= 1
        assert len(env.defenders) >= 1
        
        env.close()
    
    def test_fuel_depletion(self, env):
        """Test that fuel depletes during operation."""
        env.reset()
        
        attacker = env.attackers['a1']
        initial_fuel = attacker.fuel
        
        # Run many steps with throttle
        action = {'a1': {'throttle': 1.0, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}}
        
        for _ in range(500):
            env.step(action)
        
        final_fuel = attacker.fuel
        
        # Fuel should have decreased
        assert final_fuel < initial_fuel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
