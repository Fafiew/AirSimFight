"""
Custom RL policies for AirSimFight.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
import numpy as np
from gymnasium import spaces

try:
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    ActorCriticPolicy = object
    BaseFeaturesExtractor = object


class AirCombatFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for air combat observations.
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        # Calculate total input dimension
        total_dim = 0
        for key, subspace in observation_space.spaces.items():
            total_dim += int(np.prod(subspace.shape))
        
        # Build network
        self.net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Flatten all observations
        obs_list = []
        for key in sorted(observations.keys()):
            obs_list.append(observations[key].flatten(1))
        
        combined = torch.cat(obs_list, dim=1)
        return self.net(combined)


class AttackerPolicy(ActorCriticPolicy):
    """
    Custom policy for attacker control.
    Outputs continuous actions for throttle, pitch, yaw, roll.
    """
    
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
            # Action space: [throttle, pitch, yaw, roll]
            # We need to override this
        )


class AttackerNetwork(nn.Module):
    """
    Standalone attacker policy network (without SB3).
    """
    
    def __init__(self, obs_dim: int = 52, action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from observation."""
        return self.actor(obs)
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        return self.critic(obs)
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value.
        
        Args:
            obs: Observation tensor
            action: Action tensor (if None, sample from policy)
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        logits = self.actor(obs)
        
        # Scale to action space: [throttle: 0-1, others: -1 to 1]
        # We'll do this in the environment
        action_mean = logits
        action_std = torch.ones_like(action_mean) * 0.3
        
        if action is None:
            # Sample action
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            # Compute log prob for given action
            dist = torch.distributions.Normal(action_mean, action_std)
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        value = self.critic(obs)
        
        return action, log_prob, value


class DefenderNetwork(nn.Module):
    """
    Policy network for defender control.
    Outputs actions for scanning patterns and interceptor launching.
    """
    
    def __init__(self, obs_dim: int = 52, action_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)


def get_action_dim(observation_space: spaces.Dict) -> int:
    """
    Get total action dimension from observation space.
    
    Args:
        observation_space: Gym observation space
        
    Returns:
        Total action dimension
    """
    return 4  # throttle, pitch, yaw, roll


def get_obs_dim(observation_space: spaces.Dict) -> int:
    """
    Get total observation dimension from observation space.
    
    Args:
        observation_space: Gym observation space
        
    Returns:
        Total observation dimension
    """
    total = 0
    for key, subspace in observation_space.spaces.items():
        total += int(np.prod(subspace.shape))
    return total


def process_observation(obs, device):
    """
    Process observation dict to tensor.
    
    Args:
        obs: Observation dictionary (can be nested with agent_id keys)
        device: Torch device
        
    Returns:
        Processed observation tensor
    """
    # Handle nested observation dict: obs = {agent_id: {obs_fields}}
    # We need to flatten all values from the inner dict
    obs_list = []
    
    # Find the actual observation dict (might be nested)
    if obs and isinstance(obs, dict):
        # Check if first value is a dict (nested structure)
        first_val = next(iter(obs.values()))
        if isinstance(first_val, dict):
            # Nested: obs = {'a1': {'position': [...], 'velocity': [...]}}
            inner_obs = first_val
        else:
            # Flat: obs = {'position': [...], ...}
            inner_obs = obs
        
        # Flatten and concatenate all observations
        for key in sorted(inner_obs.keys()):
            val = inner_obs[key]
            if isinstance(val, (list, np.ndarray)):
                val = np.array(val, dtype=np.float32)
                obs_list.append(val.flatten())
    
    combined = np.concatenate(obs_list)
    tensor = torch.from_numpy(combined).float().unsqueeze(0).to(device)
    
    return tensor


def action_to_dict(action: np.ndarray) -> Dict[str, float]:
    """
    Convert flat action array to dict.
    
    Args:
        action: Action array [throttle, pitch, yaw, roll]
        
    Returns:
        Action dict
    """
    return {
        'throttle': float(np.clip(action[0], 0, 1)),
        'pitch': float(action[1]),
        'yaw': float(action[2]),
        'roll': float(action[3]),
    }


def dict_to_action(action_dict: Dict[str, float]) -> np.ndarray:
    """
    Convert action dict to flat array.
    
    Args:
        action_dict: Action dict
        
    Returns:
        Action array
    """
    return np.array([
        action_dict.get('throttle', 0.5),
        action_dict.get('pitch', 0.0),
        action_dict.get('yaw', 0.0),
        action_dict.get('roll', 0.0),
    ], dtype=np.float32)
