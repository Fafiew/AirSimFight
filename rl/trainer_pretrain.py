#!/usr/bin/env python3
"""
Attacker pretraining script.
Trains a single attacker to reach target while managing fuel.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.environment import AirCombatEnv
from sim.utils import load_config, parse_overrides, setup_logging
from rl.model_io import save_model, load_model

logger = logging.getLogger(__name__)


def setup_device(device_str: str) -> torch.device:
    """
    Setup device with CUDA fallback.
    
    Args:
        device_str: 'cpu' or 'cuda'
        
    Returns:
        Torch device
    """
    if device_str == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def print_safety_warning() -> None:
    """Print safety warning message."""
    safety_file = Path(__file__).parent.parent / "safe_use.md"
    if safety_file.exists():
        print("\n" + "="*60)
        print("⚠️  SAFETY WARNING  ⚠️")
        print("="*60)
        print("\nThis software is strictly a simulation and research tool.")
        print("Do NOT use for real-world weapons development.")
        print("Do NOT transfer models to real hardware.")
        print("\nSee safe_use.md for full guidelines.")
        print("="*60 + "\n")


def create_simple_policy(env: AirCombatEnv, device: torch.device):
    """
    Create a simple policy for pretraining.
    Uses a simple heuristic or can be replaced with actual RL.
    
    Args:
        env: Environment
        device: Torch device
        
    Returns:
        Simple policy function
    """
    # For simplicity, use a simple policy network
    from rl.policies import AttackerNetwork, process_observation, action_to_dict
    
    # Compute actual observation dimension from environment
    # Get sample observation (first agent's observation)
    sample_obs, _ = env.reset()
    first_agent_id = list(sample_obs.keys())[0]
    sample_tensor = process_observation(sample_obs[first_agent_id], device)
    obs_dim = sample_tensor.shape[1]
    
    policy = AttackerNetwork(obs_dim=obs_dim, action_dim=4).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    return policy, optimizer


def train(
    env: AirCombatEnv,
    policy,
    optimizer,
    device: torch.device,
    total_timesteps: int,
    output_dir: str,
    log_interval: int = 100
) -> None:
    """
    Train the policy.
    
    Args:
        env: Environment
        policy: Policy network
        optimizer: Optimizer
        device: Device
        total_timesteps: Total training steps
        output_dir: Output directory
        log_interval: Logging interval
    """
    from rl.policies import process_observation, action_to_dict
    
    episode_count = 0
    total_reward = 0
    episode_rewards = []
    
    obs, _ = env.reset()
    
    logger.info(f"Starting pretraining for {total_timesteps} steps")
    
    for step in range(total_timesteps):
        # Get first agent's observation (in pretrain mode, only 'a1')
        agent_id = list(obs.keys())[0]
        agent_obs = obs[agent_id]
        
        # Process observation
        obs_tensor = process_observation(agent_obs, device)
        
        # Get action from policy
        with torch.no_grad():
            action_tensor = policy(obs_tensor)
        
        action = action_to_dict(action_tensor.cpu().numpy()[0])
        
        # Step environment
        next_obs, rewards, dones, truncates, infos = env.step({agent_id: action})
        
        # Compute loss (simplified PPO-like)
        # In a full implementation, you'd use actual PPO
        reward = rewards.get(agent_id, 0)
        total_reward += reward
        
        # Simple policy gradient update
        if step % 10 == 0:
            # Dummy update for demonstration
            # Real training would use proper PPO
            pass
        
        # Check if episode done
        if dones.get('__all__', False):
            episode_count += 1
            episode_rewards.append(total_reward)
            total_reward = 0
            
            if episode_count % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-log_interval:])
                logger.info(f"Episode {episode_count}, Avg Reward: {avg_reward:.2f}")
            
            # Reset
            next_obs, _ = env.reset()
        
        obs = next_obs
        
        # Save checkpoint
        if step > 0 and step % 10000 == 0:
            save_checkpoint(policy, optimizer, output_dir, step)
    
    # Final save
    save_model(policy, output_dir, format='both')
    logger.info(f"Training complete. Model saved to {output_dir}")


def save_checkpoint(policy, optimizer, output_dir: str, step: int) -> None:
    """Save training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"checkpoint_{step}.pth")
    
    torch.save({
        'step': step,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Attacker pretraining")
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Number of training timesteps')
    parser.add_argument('--output', type=str, default='models/attacker_pretrained',
                       help='Output directory for model')
    parser.add_argument('--safety-check', action='store_true',
                       help='Print safety warning and require --force to proceed')
    parser.add_argument('--force', action='store_true',
                       help='Force run after safety check')
    parser.add_argument('--overrides', type=str, default=None,
                       help='JSON string to override config values')
    
    args = parser.parse_args()
    
    # Safety check
    if args.safety_check:
        print_safety_warning()
        if not args.force:
            print("Use --force to proceed after reading safety guidelines.")
            sys.exit(1)
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    if args.overrides:
        overrides = parse_overrides(args.overrides)
        config = {**config, **overrides}
    
    # Override timesteps if specified
    if args.timesteps:
        config.setdefault('rl', {})['total_timesteps_pretrain'] = args.timesteps
    
    total_timesteps = config.get('rl', {}).get('total_timesteps_pretrain', 1000000)
    
    # Setup logging
    log_dir = config.get('logging', {}).get('log_dir', 'logs')
    setup_logging(log_dir, config.get('logging', {}).get('tensorboard', True))
    
    # Setup device
    device = setup_device(args.device)
    
    # Create environment
    env = AirCombatEnv(
        config=config,
        device=args.device,
        headless=True,
        pretrain_mode=True
    )
    
    logger.info(f"Created pretrain environment (device={args.device})")
    
    # Create policy
    policy, optimizer = create_simple_policy(env, device)
    
    # Train
    train(
        env=env,
        policy=policy,
        optimizer=optimizer,
        device=device,
        total_timesteps=total_timesteps,
        output_dir=args.output
    )
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
