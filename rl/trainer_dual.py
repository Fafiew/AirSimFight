#!/usr/bin/env python3
"""
Dual-agent training script.
Trains attackers and defenders together with procedural episode generation.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.environment import AirCombatEnv
from sim.utils import load_config, parse_overrides, setup_logging
from rl.model_io import save_model, load_model
from rl.policies import AttackerNetwork, DefenderNetwork, process_observation, action_to_dict

logger = logging.getLogger(__name__)


def setup_device(device_str: str) -> torch.device:
    """Setup device with CUDA fallback."""
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


def create_policies(env: AirCombatEnv, device: torch.device, use_pretrained: bool = False, 
                   pretrained_path: Optional[str] = None, freeze_attacker: bool = True):
    """
    Create attacker and defender policies.
    
    Args:
        env: Environment
        device: Torch device
        use_pretrained: Whether to load pretrained attacker
        pretrained_path: Path to pretrained model
        freeze_attacker: Whether to freeze attacker weights
        
    Returns:
        Tuple of (attacker_policy, defender_policy, optimizers)
    """
    obs_dim = 52  # Approximate
    action_dim = 4
    
    # Attacker policy
    attacker_policy = AttackerNetwork(obs_dim=obs_dim, action_dim=action_dim).to(device)
    
    # Optionally load pretrained
    if use_pretrained and pretrained_path:
        logger.info(f"Loading pretrained attacker from {pretrained_path}")
        loaded = load_model(pretrained_path, device)
        if loaded is not None:
            attacker_policy.load_state_dict(loaded)
    
    # Defender policy
    defender_policy = DefenderNetwork(obs_dim=obs_dim, action_dim=action_dim).to(device)
    
    # Optimizers
    optimizer_kwargs = {'lr': 3e-4}
    
    if freeze_attacker:
        # Freeze attacker weights
        for param in attacker_policy.parameters():
            param.requires_grad = False
        attacker_optimizer = None
    else:
        attacker_optimizer = torch.optim.Adam(attacker_policy.parameters(), **optimizer_kwargs)
    
    defender_optimizer = torch.optim.Adam(defender_policy.parameters(), **optimizer_kwargs)
    
    optimizers = {
        'attacker': attacker_optimizer,
        'defender': defender_optimizer
    }
    
    return attacker_policy, defender_policy, optimizers


def train(
    env: AirCombatEnv,
    attacker_policy,
    defender_policy,
    optimizers: Dict,
    device: torch.device,
    total_timesteps: int,
    output_dir: str,
    freeze_attacker: bool = True,
    log_interval: int = 100
) -> None:
    """
    Train attacker and defender policies.
    
    Args:
        env: Environment
        attacker_policy: Attacker policy network
        defender_policy: Defender policy network  
        optimizers: Dict of optimizers
        device: Device
        total_timesteps: Total training steps
        output_dir: Output directory
        freeze_attacker: Whether to freeze attacker weights
        log_interval: Logging interval
    """
    episode_count = 0
    total_attacker_reward = 0
    total_defender_reward = 0
    episode_rewards = []
    
    # Reset environment
    obs, _ = env.reset()
    
    logger.info(f"Starting dual training for {total_timesteps} steps")
    logger.info(f"Freeze attacker: {freeze_attacker}")
    
    for step in range(total_timesteps):
        # Get actions for attackers
        attacker_actions = {}
        for attacker_id in obs.keys():
            obs_tensor = process_observation(obs[attacker_id], device)
            
            with torch.no_grad():
                action_tensor = attacker_policy(obs_tensor)
            
            attacker_actions[attacker_id] = action_to_dict(action_tensor.cpu().numpy()[0])
        
        # Step environment
        next_obs, rewards, dones, truncates, infos = env.step(attacker_actions)
        
        # Aggregate rewards
        attacker_reward = sum(r for k, r in rewards.items() if k.startswith('a'))
        total_attacker_reward += attacker_reward
        
        # Update defender (simplified)
        # In a full implementation, you'd update based on interceptor success
        
        # Check if episode done
        if dones.get('__all__', False):
            episode_count += 1
            episode_rewards.append(total_attacker_reward)
            total_attacker_reward = 0
            
            if episode_count % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-log_interval:])
                logger.info(f"Episode {episode_count}, Attacker Avg Reward: {avg_reward:.2f}")
                
                # Log procedural spawn counts
                n_att = len([k for k in env.attackers.keys() if k.startswith('a')])
                n_def = len([k for k in env.defenders.keys() if k.startswith('d')])
                logger.info(f"  Spawned: {n_att} attackers, {n_def} defenders")
            
            # Reset
            next_obs, _ = env.reset()
        
        obs = next_obs
        
        # Save checkpoint
        if step > 0 and step % 10000 == 0:
            save_checkpoint(
                attacker_policy, defender_policy, 
                optimizers, output_dir, step,
                freeze_attacker
            )
    
    # Final save
    os.makedirs(output_dir, exist_ok=True)
    save_model(attacker_policy, os.path.join(output_dir, "attacker_final"), format='both')
    save_model(defender_policy, os.path.join(output_dir, "defender_final"), format='both')
    logger.info(f"Training complete. Models saved to {output_dir}")


def save_checkpoint(attacker_policy, defender_policy, optimizers, output_dir: str, step: int, 
                   freeze_attacker: bool) -> None:
    """Save training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'defender_state_dict': defender_policy.state_dict(),
    }
    
    if not freeze_attacker and optimizers.get('attacker'):
        checkpoint['attacker_state_dict'] = attacker_policy.state_dict()
        checkpoint['attacker_optimizer_state_dict'] = optimizers['attacker'].state_dict()
    
    if optimizers.get('defender'):
        checkpoint['defender_optimizer_state_dict'] = optimizers['defender'].state_dict()
    
    checkpoint_path = os.path.join(output_dir, f"checkpoint_{step}.pth")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Dual-agent training")
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Number of training timesteps')
    parser.add_argument('--attacker-model', type=str, default=None,
                       help='Path to pretrained attacker model')
    parser.add_argument('--fine-tune-attacker', action='store_true',
                       help='Fine-tune attacker weights (default: freeze)')
    parser.add_argument('--output', type=str, default='models/dual_train',
                       help='Output directory for models')
    parser.add_argument('--safety-check', action='store_true',
                       help='Print safety warning')
    parser.add_argument('--force', action='store_true',
                       help='Force run after safety check')
    parser.add_argument('--overrides', type=str, default=None,
                       help='JSON string to override config')
    
    args = parser.parse_args()
    
    # Safety check
    if args.safety_check:
        print_safety_warning()
        if not args.force:
            print("Use --force to proceed.")
            sys.exit(1)
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    if args.overrides:
        overrides = parse_overrides(args.overrides)
        config = {**config, **overrides}
    
    # Override timesteps
    if args.timesteps:
        config.setdefault('rl', {})['total_timesteps_dual'] = args.timesteps
    
    total_timesteps = config.get('rl', {}).get('total_timesteps_dual', 2000000)
    
    # Setup logging
    log_dir = config.get('logging', {}).get('log_dir', 'logs')
    setup_logging(log_dir, config.get('logging', {}).get('tensorboard', True))
    
    # Setup device
    device = setup_device(args.device)
    
    # Create environment (dual training mode)
    env = AirCombatEnv(
        config=config,
        device=args.device,
        headless=True,
        pretrain_mode=False  # Dual training
    )
    
    logger.info(f"Created dual training environment (device={args.device})")
    
    # Create policies
    use_pretrained = args.attacker_model is not None
    freeze_attacker = not args.fine_tune_attacker
    
    attacker_policy, defender_policy, optimizers = create_policies(
        env, device, 
        use_pretrained=use_pretrained,
        pretrained_path=args.attacker_model,
        freeze_attacker=freeze_attacker
    )
    
    # Train
    train(
        env=env,
        attacker_policy=attacker_policy,
        defender_policy=defender_policy,
        optimizers=optimizers,
        device=device,
        total_timesteps=total_timesteps,
        output_dir=args.output,
        freeze_attacker=freeze_attacker
    )
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
