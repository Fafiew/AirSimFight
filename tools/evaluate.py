#!/usr/bin/env python3
"""
Evaluation script for trained models.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.environment import AirCombatEnv
from sim.utils import load_config, setup_logging
from rl.policies import AttackerNetwork, process_observation, action_to_dict
from rl.model_io import load_model

logger = logging.getLogger(__name__)


def evaluate_model(
    env: AirCombatEnv,
    policy,
    device: torch.device,
    num_episodes: int = 10,
    render: bool = False
) -> dict:
    """
    Evaluate a trained model.
    
    Args:
        env: Environment
        policy: Policy network
        device: Device
        num_episodes: Number of evaluation episodes
        render: Whether to render
        
    Returns:
        Evaluation metrics dict
    """
    episode_rewards = []
    episode_lengths = []
    successes = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Get action
            attacker_ids = list(obs.keys())
            if not attacker_ids:
                break
            
            attacker_id = attacker_ids[0]
            obs_tensor = process_observation(obs[attacker_id], device)
            
            with torch.no_grad():
                action_tensor = policy(obs_tensor)
            
            action = action_to_dict(action_tensor.cpu().numpy()[0])
            
            # Step
            next_obs, rewards, dones, truncates, infos = env.step({attacker_id: action})
            
            reward = rewards.get(attacker_id, 0)
            episode_reward += reward
            episode_length += 1
            
            # Render
            if render:
                env.render()
            
            # Check done
            if dones.get('__all__', False):
                break
            
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Success = positive reward (reached target)
        if episode_reward > 0:
            successes += 1
        
        logger.info(f"Episode {episode + 1}: reward={episode_reward:.2f}, length={episode_length}")
    
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': successes / num_episodes,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file (.pth)')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Config file path')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render during evaluation')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    
    # Create environment
    env = AirCombatEnv(
        config=config,
        device=args.device,
        headless=not args.render,
        pretrain_mode=True
    )
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    state_dict = load_model(args.model, device)
    
    if state_dict is None:
        logger.error("Failed to load model")
        sys.exit(1)
    
    # Create policy
    policy = AttackerNetwork(obs_dim=52, action_dim=4).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    
    # Evaluate
    logger.info(f"Evaluating for {args.episodes} episodes")
    metrics = evaluate_model(env, policy, device, args.episodes, args.render)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Mean Length: {metrics['mean_length']:.1f}")
    print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
    print("="*50)
    
    env.close()


if __name__ == "__main__":
    main()
