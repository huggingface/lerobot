#!/usr/bin/env python3
"""
Evaluation Script for π*₀.₆ RECAP

Supports:
- Dataset evaluation
- Environment rollout (if env provided)
- Metrics logging

Usage:
    python eval_pi_star_recap.py \
        --checkpoint path/to/checkpoint.pt \
        --dataset_path path/to/dataset \
        --output_dir ./eval_results
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi_star_recap import PiStarRECAPConfig, PiStarRECAPPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate π*₀.₆ RECAP")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    
    # Evaluation settings
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    
    # Rollout settings (optional)
    parser.add_argument("--env_type", type=str, default=None, help="Environment type for rollout")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes for rollout")
    parser.add_argument("--max_steps_per_episode", type=int, default=500)
    
    # Visualization
    parser.add_argument("--save_videos", action="store_true", help="Save rollout videos")
    
    return parser.parse_args()


def evaluate_on_dataset(
    policy: PiStarRECAPPolicy,
    dataloader: DataLoader,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate on dataset"""
    policy.eval()
    
    all_metrics = {
        'total_loss': [],
        'v_loss': [],
        'q_loss': [],
        'policy_loss': [],
        'q_values': [],
        'v_values': [],
        'advantages': [],
    }
    
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Compute losses
            losses = policy.compute_loss(batch)
            
            all_metrics['total_loss'].append(losses['loss'].item())
            all_metrics['v_loss'].append(losses['v_loss'].item())
            all_metrics['q_loss'].append(losses['q_loss'].item())
            all_metrics['policy_loss'].append(losses['policy_loss'].item())
            
            # Compute Q, V, Advantage for analysis
            images = batch.get('observation.images')
            actions = batch['action']
            
            context = policy._get_vlm_features(images=images)
            
            q_values = torch.stack([q(context, actions) for q in policy.q_networks])
            q_min = q_values.min(dim=0)[0]
            v_value = policy.v_network(context)
            advantage = q_min - v_value
            
            all_metrics['q_values'].append(q_min.mean().item())
            all_metrics['v_values'].append(v_value.mean().item())
            all_metrics['advantages'].append(advantage.mean().item())
            
            num_samples += actions.shape[0]
            
            if max_samples and num_samples >= max_samples:
                break
    
    # Aggregate
    results = {k: np.mean(v) for k, v in all_metrics.items()}
    results['num_samples'] = num_samples
    
    return results


def rollout_in_env(
    policy: PiStarRECAPPolicy,
    env,
    num_episodes: int,
    max_steps: int,
    save_videos: bool = False,
):
    """Rollout policy in environment"""
    policy.eval()
    
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        frames = [] if save_videos else None
        
        for step in range(max_steps):
            # Prepare observation
            batch = {
                'observation.images': torch.from_numpy(obs['images']).unsqueeze(0).cuda(),
                'observation.state': torch.from_numpy(obs['state']).unsqueeze(0).cuda(),
            }
            
            # Select action
            with torch.no_grad():
                action = policy.select_action(batch)
                action = action.cpu().numpy()[0]
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if save_videos:
                frames.append(env.render(mode='rgb_array'))
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {ep+1}: reward={episode_reward:.2f}, length={episode_length}")
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean([r > 0 for r in episode_rewards]),
    }
    
    return results


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cuda')
    config = PiStarRECAPConfig.from_dict(checkpoint['config'])
    
    # Create policy
    policy = PiStarRECAPPolicy(config)
    policy.load_checkpoint(args.checkpoint)
    policy = policy.cuda()
    
    # Evaluate on dataset
    print(f"\nEvaluating on dataset: {args.dataset_path}")
    dataset = LeRobotDataset(args.dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    
    dataset_results = evaluate_on_dataset(policy, dataloader, args.max_samples)
    
    print("\nDataset Evaluation Results:")
    for key, value in dataset_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'dataset_path': args.dataset_path,
        'dataset_evaluation': dataset_results,
    }
    
    # Environment rollout (if specified)
    if args.env_type:
        print(f"\nRolling out in environment: {args.env_type}")
        # This would require environment setup
        # from lerobot.envs import make_env
        # env = make_env(args.env_type)
        # env_results = rollout_in_env(
        #     policy, env, args.num_episodes, 
        #     args.max_steps_per_episode, args.save_videos
        # )
        # results['environment_evaluation'] = env_results
        pass
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
