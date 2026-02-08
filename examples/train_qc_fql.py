#!/usr/bin/env python
"""
Training script for QC-FQL (Q-Chunking with Fitted Q-Learning).

This script demonstrates how to train QC-FQL on a LeRobot dataset for
offline-to-online RL on long-horizon manipulation tasks.

Usage:
    python train_qc_fql.py \
        --dataset_path path/to/dataset \
        --output_dir ./checkpoints/qc_fql \
        --action_chunk_size 4 \
        --num_critics 10

Reference:
    Li et al., "Reinforcement Learning with Action Chunking", 2025
    https://arxiv.org/abs/2507.07969
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.qc_fql import QCFQLPolicy, QCFQLConfig, QCFQLProcessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train QC-FQL policy")
    
    # Dataset
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to LeRobot dataset")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    
    # Model
    parser.add_argument("--action_chunk_size", type=int, default=4, help="Action chunk size (k)")
    parser.add_argument("--num_critics", type=int, default=10, help="Number of critics in ensemble")
    parser.add_argument("--distillation_weight", type=float, default=1.0, help="Distillation loss weight (lambda)")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor")
    
    # Training
    parser.add_argument("--num_pretrain_steps", type=int, default=100000, help="Offline pretraining steps")
    parser.add_argument("--num_online_steps", type=int, default=100000, help="Online training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--update_to_data_ratio", type=int, default=1, help="UTD ratio")
    
    # Logging
    parser.add_argument("--output_dir", type=str, default="./checkpoints/qc_fql", help="Output directory")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=10000, help="Checkpoint save interval")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()


def create_dataset(dataset_path: str, batch_size: int):
    """Create LeRobot dataset and dataloader."""
    dataset = LeRobotDataset(dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return dataset, dataloader


def pretrain_behavior_policy(
    policy: QCFQLPolicy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    log_interval: int,
    writer: SummaryWriter,
    device: str,
):
    """
    Pretrain behavior policy using flow matching.
    
    This phase only trains the flow-matching behavior policy to capture
    the distribution of offline data.
    """
    logger.info(f"Pretraining behavior policy for {num_steps} steps...")
    
    policy.train()
    step = 0
    epoch = 0
    
    pbar = tqdm(total=num_steps, desc="Pretraining behavior policy")
    
    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break
            
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Compute loss
            loss = policy.compute_loss_behavior(batch)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.behavior_policy.parameters(), 1.0)
            optimizer.step()
            
            # Logging
            if step % log_interval == 0:
                writer.add_scalar("pretrain/behavior_loss", loss.item(), step)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            step += 1
            pbar.update(1)
        
        epoch += 1
    
    pbar.close()
    logger.info("Behavior policy pretraining complete!")


def train_qc_fql(
    policy: QCFQLPolicy,
    dataloader: DataLoader,
    optimizers: dict[str, torch.optim.Optimizer],
    num_steps: int,
    log_interval: int,
    save_interval: int,
    output_dir: str,
    writer: SummaryWriter,
    device: str,
):
    """
    Main training loop for QC-FQL.
    
    Alternates between:
    1. Training behavior policy (flow matching)
    2. Training critic (TD learning)
    3. Training policy (Q-maximization + distillation)
    """
    logger.info(f"Training QC-FQL for {num_steps} steps...")
    
    policy.train()
    step = 0
    epoch = 0
    
    pbar = tqdm(total=num_steps, desc="Training QC-FQL")
    
    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break
            
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Update UTD (Update-To-Data) ratio
            for _ in range(policy.config.update_to_data_ratio):
                # 1. Train behavior policy
                loss_behavior = policy.compute_loss_behavior(batch)
                optimizers["behavior"].zero_grad()
                loss_behavior.backward()
                torch.nn.utils.clip_grad_norm_(policy.behavior_policy.parameters(), 1.0)
                optimizers["behavior"].step()
                
                # 2. Train critic
                loss_critic = policy.compute_loss_critic(batch)
                optimizers["critic"].zero_grad()
                loss_critic.backward()
                torch.nn.utils.clip_grad_norm_(policy.critic_ensemble.parameters(), 1.0)
                optimizers["critic"].step()
                
                # Update target networks
                policy.update_target_networks()
                
                # 3. Train policy
                loss_policy = policy.compute_loss_policy(batch)
                optimizers["policy"].zero_grad()
                loss_policy.backward()
                torch.nn.utils.clip_grad_norm_(policy.policy.parameters(), 1.0)
                optimizers["policy"].step()
            
            # Logging
            if step % log_interval == 0:
                writer.add_scalar("train/behavior_loss", loss_behavior.item(), step)
                writer.add_scalar("train/critic_loss", loss_critic.item(), step)
                writer.add_scalar("train/policy_loss", loss_policy.item(), step)
                
                # Log Q-values
                with torch.no_grad():
                    state = batch["observation.state"]
                    action_chunk = batch["action"]
                    q_values = policy.critic_ensemble(state, action_chunk)
                    writer.add_scalar("train/mean_q", q_values.mean().item(), step)
                    writer.add_scalar("train/min_q", q_values.min().item(), step)
                
                pbar.set_postfix({
                    "behavior": f"{loss_behavior.item():.3f}",
                    "critic": f"{loss_critic.item():.3f}",
                    "policy": f"{loss_policy.item():.3f}",
                })
            
            # Save checkpoint
            if step % save_interval == 0 and step > 0:
                checkpoint_path = Path(output_dir) / f"checkpoint_{step}.pt"
                torch.save({
                    "step": step,
                    "policy_state_dict": policy.state_dict(),
                    "optimizers_state_dict": {k: v.state_dict() for k, v in optimizers.items()},
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            step += 1
            pbar.update(1)
        
        epoch += 1
    
    pbar.close()
    
    # Save final model
    final_path = Path(output_dir) / "final_model.pt"
    torch.save({
        "step": step,
        "policy_state_dict": policy.state_dict(),
        "config": policy.config,
    }, final_path)
    logger.info(f"Saved final model to {final_path}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=output_dir / "logs")
    
    # Create dataset
    logger.info(f"Loading dataset from {args.dataset_path}...")
    dataset, dataloader = create_dataset(args.dataset_path, args.batch_size)
    
    # Infer dimensions from dataset
    state_dim = dataset.features["observation.state"].shape[0]
    action_dim = dataset.features["action"].shape[0]
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Create config
    config = QCFQLConfig(
        action_chunk_size=args.action_chunk_size,
        num_critics=args.num_critics,
        distillation_weight=args.distillation_weight,
        discount=args.discount,
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=args.batch_size,
        num_pretrain_steps=args.num_pretrain_steps,
        num_online_steps=args.num_online_steps,
        learning_rate=args.learning_rate,
        update_to_data_ratio=args.update_to_data_ratio,
        device=args.device,
    )
    
    # Create policy
    logger.info("Creating QC-FQL policy...")
    policy = QCFQLPolicy(config)
    policy.to(args.device)
    
    # Create optimizers
    optimizers = {
        "behavior": torch.optim.Adam(
            policy.behavior_policy.parameters(),
            lr=config.behavior_policy_learning_rate,
        ),
        "critic": torch.optim.Adam(
            policy.critic_ensemble.parameters(),
            lr=config.critic_learning_rate,
        ),
        "policy": torch.optim.Adam(
            policy.policy.parameters(),
            lr=config.actor_learning_rate,
        ),
    }
    
    # Phase 1: Pretrain behavior policy
    pretrain_behavior_policy(
        policy=policy,
        dataloader=dataloader,
        optimizer=optimizers["behavior"],
        num_steps=config.num_pretrain_steps,
        log_interval=args.log_interval,
        writer=writer,
        device=args.device,
    )
    
    # Phase 2: Train full QC-FQL
    train_qc_fql(
        policy=policy,
        dataloader=dataloader,
        optimizers=optimizers,
        num_steps=config.num_online_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        output_dir=str(output_dir),
        writer=writer,
        device=args.device,
    )
    
    writer.close()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
