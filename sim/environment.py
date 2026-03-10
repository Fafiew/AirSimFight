"""
Air Combat Gym-style environment.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Any, Tuple
import logging

from sim.physics import PhysicsWorld
from sim.entities import Entity, Attacker, Defender, Interceptor, Target
from sim.sensors import Radar, InfraredSeeker
from sim.utils import sample_in_ring, distance

logger = logging.getLogger(__name__)


class AirCombatEnv(gym.Env):
    """
    Gym-style air combat environment.
    Supports multi-agent training with procedural episode generation.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        config: dict,
        scenario_json: Optional[str] = None,
        device: str = "cpu",
        headless: bool = True,
        pretrain_mode: bool = False
    ):
        """
        Initialize environment.
        
        Args:
            config: Configuration dictionary
            scenario_json: Optional path to visualization JSON
            device: Device for computations (cpu/cuda)
            headless: If True, use headless physics
            pretrain_mode: If True, disable defenders (single-agent pretraining)
        """
        super().__init__()
        
        self.config = config
        self.scenario_json = scenario_json
        self.device = device
        self.headless = headless
        self.pretrain_mode = pretrain_mode
        
        # Initialize physics
        self.physics_world = PhysicsWorld(config, headless=headless)
        
        # Entity storage
        self.attackers: Dict[str, Attacker] = {}
        self.defenders: Dict[str, Defender] = {}
        self.interceptors: Dict[str, Interceptor] = {}
        self.targets: Dict[str, Target] = {}
        
        # All entities for sensor processing
        self.entities: Dict[str, Entity] = {}
        
        # Sensor configs
        radar_cfg = config.get('sensors', {}).get('radar', {})
        ir_cfg = config.get('sensors', {}).get('ir', {})
        
        # Action and observation spaces will be set in reset
        self.action_space = spaces.Dict({
            'throttle': spaces.Box(0, 1, dtype=np.float32),
            'pitch': spaces.Box(-1, 1, dtype=np.float32),
            'yaw': spaces.Box(-1, 1, dtype=np.float32),
            'roll': spaces.Box(-1, 1, dtype=np.float32),
        })
        
        self.observation_space = spaces.Dict({
            'position': spaces.Box(-20000, 20000, (3,), dtype=np.float32),
            'velocity': spaces.Box(-1000, 1000, (3,), dtype=np.float32),
            'fuel': spaces.Box(0, 100, (1,), dtype=np.float32),
            'health': spaces.Box(0, 100, (1,), dtype=np.float32),
            'radar_detections': spaces.Box(0, 1, (10, 6), dtype=np.float32),  # max 10 detections
            'ir_signal': spaces.Box(0, 1, (4,), dtype=np.float32),
        })
        
        # Tracking
        self.step_count = 0
        self.max_steps = 1000
        
        # Reward config
        self.target_reward = config.get('rl', {}).get('attacker_rewards', {}).get('target_reward', 100.0)
        self.destroyed_penalty = config.get('rl', {}).get('attacker_rewards', {}).get('destroyed_penalty', -100.0)
        self.fuel_cost_coeff = config.get('rl', {}).get('attacker_rewards', {}).get('fuel_cost_coeff', 50.0)
        
        # Load scenario if provided
        if scenario_json and not headless:
            self._load_visualization_scenario(scenario_json)
        
        logger.info(f"AirCombatEnv initialized (headless={headless}, pretrain={pretrain_mode})")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed
            options: Optional options dict
            
        Returns:
            Tuple of (observations, info)
        """
        super().reset(seed=seed)
        
        # Clear existing entities
        self._clear_entities()
        
        # Reset physics
        self.physics_world.reset()
        self.step_count = 0
        
        # Generate new episode
        if self.pretrain_mode:
            self._generate_pretrain_episode()
        else:
            self._generate_procedural_episode()
        
        # Get initial observations
        observations = self._get_observations()
        
        return observations, {}
    
    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one step of the environment.
        
        Args:
            actions: Dict of agent_id -> action
            
        Returns:
            Tuple of (observations, rewards, dones, truncates, infos)
        """
        self.step_count += 1
        
        # Process attacker actions
        for attacker_id, action in actions.items():
            if attacker_id in self.attackers:
                attacker = self.attackers[attacker_id]
                if attacker.is_alive:
                    attacker.step(action, self.config.get('sim', {}).get('timestep', 0.02), self.physics_world)
        
        # Step physics simulation
        dt = self.config.get('sim', {}).get('timestep', 0.02)
        self.physics_world.step(dt)
        
        # Sync positions from physics to entities
        for attacker_id, attacker in self.attackers.items():
            if attacker.is_alive and attacker.body_id is not None:
                attacker.position = self.physics_world.get_position(attacker.body_id)
                attacker.velocity = self.physics_world.get_velocity(attacker.body_id)
        
        # Process defender actions (simplified - just scan)
        for defender_id, defender in self.defenders.items():
            if defender.is_alive:
                # Simple defender AI - launch at nearest attacker if in range
                detections = defender.scan(self)
                if detections:
                    # Find nearest attacker
                    nearest = min(detections, key=lambda d: d['distance'])
                    if np.random.random() < 0.1:  # 10% chance to launch
                        defender.launch_interceptor(nearest['id'], self.physics_world)
        
        # Update interceptors
        self._update_interceptors()
        
        # Check collisions and remove destroyed entities
        self._check_collisions()
        
        # Check boundary timeout
        self._check_boundaries()
        
        # Get observations, rewards, dones
        observations = self._get_observations()
        rewards = self._compute_rewards()
        dones = self._get_dones()
        infos = self._get_infos()
        
        # Truncation
        truncates = {'__all__': self.step_count >= self.max_steps}
        
        return observations, rewards, dones, truncates, infos
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Render mode ('human' or 'rgb_array')
            
        Returns:
            RGB array if mode is 'rgb_array', else None
        """
        if mode == 'human':
            # Visualization handled by viz module
            return None
        elif mode == 'rgb_array':
            # Could implement screenshot capture here
            return None
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        self.physics_world.close()
        logger.info("AirCombatEnv closed")
    
    def _clear_entities(self) -> None:
        """Clear all entities."""
        for attacker in self.attackers.values():
            if attacker.body_id is not None:
                self.physics_world.remove_entity(attacker.id)
        
        for defender in self.defenders.values():
            if defender.body_id is not None:
                self.physics_world.remove_entity(defender.id)
        
        for interceptor in self.interceptors.values():
            if interceptor.body_id is not None:
                self.physics_world.remove_entity(interceptor.id)
        
        for target in self.targets.values():
            if target.body_id is not None:
                self.physics_world.remove_entity(target.id)
        
        self.attackers.clear()
        self.defenders.clear()
        self.interceptors.clear()
        self.targets.clear()
        self.entities.clear()
    
    def _generate_pretrain_episode(self) -> None:
        """Generate single-agent pretrain episode."""
        # Single attacker, no defenders
        # Sample position in training corridor
        spawn_cfg = self.config.get('procedural_spawn', {})
        
        # Random position in outer ring
        position = sample_in_ring(
            spawn_cfg.get('R_att_min', 5000),
            spawn_cfg.get('R_att_max', 20000),
            spawn_cfg.get('H_att_min', 1500),
            spawn_cfg.get('H_att_max', 5000)
        )
        
        # Create attacker
        attacker = Attacker('a1', position, self.config)
        
        # Add to physics
        body_id = self.physics_world.add_entity(
            attacker.id,
            mass=attacker.mass_current,
            position=attacker.position,
            radius=2.0
        )
        attacker.body_id = body_id
        
        # Setup radar for attacker
        radar_cfg = self.config.get('sensors', {}).get('radar', {})
        attacker.radar = Radar(
            range=radar_cfg.get('default_range', 15000),
            h_fov_deg=radar_cfg.get('horizontal_fov_deg', 120),
            v_fov_deg=radar_cfg.get('vertical_fov_deg', 30),
            rcs_noise_std=radar_cfg.get('rcs_noise_std', 0.1)
        )
        
        ir_cfg = self.config.get('sensors', {}).get('ir', {})
        attacker.ir = InfraredSeeker(
            range=ir_cfg.get('default_range', 5000),
            lock_angle_deg=ir_cfg.get('lock_angle_deg', 10),
            noise_std=ir_cfg.get('noise_std', 0.05)
        )
        
        self.attackers[attacker.id] = attacker
        self.entities[attacker.id] = attacker
        
        # Single target at origin
        target = Target('t1', np.array([0, 0, 0], dtype=np.float32))
        
        # Add target to physics
        target_body_id = self.physics_world.add_entity(
            target.id,
            mass=0.1,
            position=target.position,
            radius=5.0
        )
        target.body_id = target_body_id
        
        self.targets[target.id] = target
        self.entities[target.id] = target
    
    def _generate_procedural_episode(self) -> None:
        """Generate dual-agent procedural episode."""
        spawn_cfg = self.config.get('procedural_spawn', {})
        
        # Sample number of attackers and defenders (1-10 each)
        n_attackers = np.random.randint(1, 11)
        n_defenders = np.random.randint(1, 11)
        
        logger.debug(f"Generating episode: {n_attackers} attackers, {n_defenders} defenders")
        
        # Target at origin
        target = Target('t1', np.array([0, 0, 0], dtype=np.float32))
        self.targets[target.id] = target
        self.entities[target.id] = target
        
        # Create defenders in inner ring
        for i in range(n_defenders):
            defender_id = f'd{i+1}'
            position = sample_in_ring(
                spawn_cfg.get('R_def_min', 50),
                spawn_cfg.get('R_def_max', 500),
                0, 50  # Near ground
            )
            position[2] = 0  # Force to ground
            
            defender = Defender(defender_id, position, self.config)
            
            # Add to physics
            body_id = self.physics_world.add_entity(
                defender.id,
                mass=1000.0,
                position=defender.position,
                radius=5.0
            )
            defender.body_id = body_id
            
            # Setup radar
            radar_cfg = self.config.get('sensors', {}).get('radar', {})
            defender.radar_sensor = Radar(
                range=radar_cfg.get('default_range', 15000),
                h_fov_deg=radar_cfg.get('horizontal_fov_deg', 120),
                v_fov_deg=radar_cfg.get('vertical_fov_deg', 30),
                rcs_noise_std=radar_cfg.get('rcs_noise_std', 0.1)
            )
            
            self.defenders[defender.id] = defender
            self.entities[defender.id] = defender
        
        # Create attackers in outer ring
        for i in range(n_attackers):
            attacker_id = f'a{i+1}'
            position = sample_in_ring(
                spawn_cfg.get('R_att_min', 5000),
                spawn_cfg.get('R_att_max', 20000),
                spawn_cfg.get('H_att_min', 1500),
                spawn_cfg.get('H_att_max', 5000)
            )
            
            attacker = Attacker(attacker_id, position, self.config)
            
            # Add to physics
            body_id = self.physics_world.add_entity(
                attacker.id,
                mass=attacker.mass_current,
                position=attacker.position,
                radius=2.0
            )
            attacker.body_id = body_id
            
            # Setup sensors
            radar_cfg = self.config.get('sensors', {}).get('radar', {})
            attacker.radar = Radar(
                range=radar_cfg.get('default_range', 15000),
                h_fov_deg=radar_cfg.get('horizontal_fov_deg', 120),
                v_fov_deg=radar_cfg.get('vertical_fov_deg', 30),
                rcs_noise_std=radar_cfg.get('rcs_noise_std', 0.1)
            )
            
            ir_cfg = self.config.get('sensors', {}).get('ir', {})
            attacker.ir = InfraredSeeker(
                range=ir_cfg.get('default_range', 5000),
                lock_angle_deg=ir_cfg.get('lock_angle_deg', 10),
                noise_std=ir_cfg.get('noise_std', 0.05)
            )
            
            self.attackers[attacker.id] = attacker
            self.entities[attacker.id] = attacker
    
    def _load_visualization_scenario(self, path: str) -> None:
        """Load visualization scenario (for display only, not training)."""
        from sim.json_loader import load_scenario
        
        scenario = load_scenario(path)
        
        # Only create entities for visualization - don't setup physics
        for pos_data in scenario['attackers']:
            attacker = Attacker(pos_data.id, np.array(pos_data.position), self.config)
            self.attackers[attacker.id] = attacker
            self.entities[attacker.id] = attacker
        
        for pos_data in scenario['defenders']:
            defender = Defender(pos_data.id, np.array(pos_data.position), self.config)
            self.defenders[defender.id] = defender
            self.entities[defender.id] = defender
        
        for pos_data in scenario['targets']:
            target = Target(pos_data.id, np.array(pos_data.position), value=100.0)
            self.targets[target.id] = target
            self.entities[target.id] = target
    
    def _update_interceptors(self) -> None:
        """Update all active interceptors."""
        dt = self.config.get('sim', {}).get('timestep', 0.02)
        
        for interceptor_id, interceptor in list(self.interceptors.items()):
            if not interceptor.is_active:
                continue
            
            # Find target
            target = self.attackers.get(interceptor.target_id)
            if target is None or not target.is_alive:
                interceptor.is_active = False
                continue
            
            # Update interceptor
            interceptor.update(dt, self.physics_world, target.position)
    
    def _check_collisions(self) -> None:
        """Check for collisions between interceptors and attackers."""
        for interceptor_id, interceptor in list(self.interceptors.items()):
            if not interceptor.is_active:
                continue
            
            # Check collision with target
            target = self.attackers.get(interceptor.target_id)
            if target and target.is_alive:
                dist = distance(interceptor.position, target.position)
                if dist < 10:  # Hit radius
                    target.apply_damage(100)  # Destroy attacker
                    interceptor.is_active = False
                    logger.info(f"Interceptor {interceptor_id} destroyed attacker {target.id}")
    
    def _check_boundaries(self) -> None:
        """Check for entities that timed out at boundary."""
        world_size = self.config.get('sim', {}).get('world_size', 20000)
        
        # Check attackers
        for attacker in self.attackers.values():
            if not attacker.is_alive:
                continue
            
            dist = distance(attacker.position, np.zeros(3))
            if dist > world_size:
                timeout = self.config.get('sim', {}).get('boundary_timeout_seconds', 5)
                # In a full implementation, we'd track time at boundary
                # For now, just let them fly back
    
    def _get_observations(self) -> Dict:
        """Get observations for all agents."""
        obs = {}
        
        # Attacker observations
        for attacker_id, attacker in self.attackers.items():
            if not attacker.is_alive:
                continue
            
            # Basic state
            attacker_obs = {
                'position': attacker.position,
                'velocity': attacker.velocity,
                'fuel': np.array([attacker.fuel]),
                'health': np.array([attacker.health]),
            }
            
            # Radar detections
            radar_detections = np.zeros((10, 6), dtype=np.float32)
            if hasattr(attacker, 'radar') and attacker.radar:
                detections = attacker.radar.scan(attacker, self.physics_world, list(self.entities.values()))
                for i, det in enumerate(detections[:10]):
                    radar_detections[i] = [
                        det.get('distance', 0) / 15000,
                        det.get('bearing', 0) / np.pi,
                        det.get('rcs', 0),
                        det.get('confidence', 0),
                        det['position'][0] / 20000,
                        det['position'][1] / 20000,
                    ]
            attacker_obs['radar_detections'] = radar_detections
            
            # IR signal
            ir_signal = np.zeros((4,), dtype=np.float32)
            if hasattr(attacker, 'ir') and attacker.ir:
                ir = attacker.ir.get_signal(attacker, self.physics_world, list(self.entities.values()))
                if ir.get('locked') and ir.get('target'):
                    t = ir['target']
                    ir_signal = np.array([
                        t['distance'] / 5000,
                        t['angle'] / np.pi,
                        t['signal_strength'],
                        t['confidence']
                    ], dtype=np.float32)
            attacker_obs['ir_signal'] = ir_signal
            
            obs[attacker_id] = attacker_obs
        
        return obs
    
    def _compute_rewards(self) -> Dict:
        """Compute rewards for all agents."""
        rewards = {}
        
        # Attacker rewards
        for attacker_id, attacker in self.attackers.items():
            reward = 0.0
            
            if not attacker.is_alive:
                reward += self.destroyed_penalty
            else:
                # Distance to target reward
                target = list(self.targets.values())[0] if self.targets else None
                if target:
                    dist = distance(attacker.position, target.position)
                    # Small reward for getting closer
                    reward += -0.1 * dist / 1000
                    
                    # Target reached
                    if dist < 50:
                        reward += self.target_reward
            
            # Fuel cost
            reward -= self.fuel_cost_coeff * (attacker.fuel_capacity - attacker.fuel) / attacker.fuel_capacity
            
            rewards[attacker_id] = reward
        
        return rewards
    
    def _get_dones(self) -> Dict:
        """Get done flags for all agents."""
        dones = {'__all__': False}
        
        # Episode ends when all attackers dead or target reached
        all_dead = all(not a.is_alive for a in self.attackers.values())
        
        # Check if target reached
        target_reached = False
        for attacker in self.attackers.values():
            if attacker.is_alive:
                target = list(self.targets.values())[0] if self.targets else None
                if target and distance(attacker.position, target.position) < 50:
                    target_reached = True
        
        if all_dead or target_reached or self.step_count >= self.max_steps:
            dones['__all__'] = True
        
        return dones
    
    def _get_infos(self) -> Dict:
        """Get info dicts for all agents."""
        infos = {}
        
        for attacker_id, attacker in self.attackers.items():
            infos[attacker_id] = {
                'is_alive': attacker.is_alive,
                'fuel': attacker.fuel,
                'health': attacker.health,
                'position': attacker.position.tolist(),
            }
        
        return infos
