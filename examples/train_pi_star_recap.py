#!/usr/bin/env python3
"""
Production Training Script for π*₀.₆ RECAP

Features:
- Distributed training (FSDP/DDP)
- Mixed precision training
- Checkpoint management
- Wandb logging
- Evaluation during training
- Automatic resume

Usage:
    # Single GPU
    python train_pi_star_recap.py --config configs/pi_star_recap.yaml
    
    # Multi-GPU with FSDP
    torchrun --nproc_per_node=4 train_pi_star_recap.py --config configs/pi_star_recap.yaml --use_fsdp
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi_star_recap import PiStarRECAPConfig, PiStarRECAPPolicy
from lerobot.policies.pi_star_recap.distributed import DistributedManager


def parse_args():
    parser = argparse.ArgumentParser(description="Train π*₀.₆ RECAP")
    
    # Config
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="./outputs/pi_star_recap")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=None)
    
    # Distributed
    parser.add_argument("--use_fsdp", action="store_true", help="Use FSDP")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="pi-star-recap")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=5000)
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    
    return parser.parse_args()


def setup_logging(args, config, distributed_manager):
    """Setup logging"""
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup wandb
    if args.use_wandb and WANDB_AVAILABLE and distributed_manager.is_main_process():
        wandb.init(
            project=args.wandb_project,
            name=f"pi_star_recap_{time.strftime('%Y%m%d_%H%M%S')}",
            config=config.to_dict(),
        )
    
    return log_dir


def create_dataloader(
    dataset_path: str,
    batch_size: int,
    num_workers: int,
    distributed_manager: DistributedManager,
) -> DataLoader:
    """Create dataloader with distributed sampler"""
    dataset = LeRobotDataset(dataset_path)
    
    sampler = None
    if distributed_manager.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed_manager.world_size,
            rank=distributed_manager.rank,
            shuffle=True,
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader


def train_epoch(
    policy: PiStarRECAPPolicy,
    dataloader: DataLoader,
    epoch: int,
    args,
    distributed_manager: DistributedManager,
):
    """Train for one epoch"""
    policy.train()
    
    if hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)
    
    epoch_metrics = {
        'loss': [],
        'v_loss': [],
        'q_loss': [],
        'policy_loss': [],
    }
    
    for step, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Training step
        metrics = policy.training_step(batch)
        
        # Accumulate metrics
        for key in epoch_metrics:
            if key in metrics:
                epoch_metrics[key].append(metrics[key])
        
        # Logging
        global_step = policy.global_step
        if global_step % args.log_interval == 0:
            if distributed_manager.is_main_process():
                log_str = f"Epoch {epoch} Step {global_step}: "
                log_str += f"loss={metrics['loss']:.4f} "
                log_str += f"v={metrics['v_loss']:.4f} "
                log_str += f"q={metrics['q_loss']:.4f} "
                log_str += f"pi={metrics['policy_loss']:.4f} "
                log_str += f"lr={metrics['lr']:.6f}"
                print(log_str)
                
                # Wandb logging
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'train/loss': metrics['loss'],
                        'train/v_loss': metrics['v_loss'],
                        'train/q_loss': metrics['q_loss'],
                        'train/policy_loss': metrics['policy_loss'],
                        'train/lr': metrics['lr'],
                        'train/step': global_step,
                    })
        
        # Save checkpoint
        if global_step % args.save_interval == 0:
            if distributed_manager.is_main_process():
                checkpoint_path = Path(args.output_dir) / f"checkpoint_{global_step}.pt"
                policy.save_checkpoint(
                    str(checkpoint_path),
                    metadata={'epoch': epoch, 'step': global_step}
                )
        
        # Check max steps
        if args.max_steps and global_step >= args.max_steps:
            break
    
    # Aggregate epoch metrics
    avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items() if v}
    return avg_metrics


def evaluate(
    policy: PiStarRECAPPolicy,
    dataloader: DataLoader,
    distributed_manager: DistributedManager,
) -> dict:
    """Evaluate policy"""
    policy.eval()
    
    all_metrics = []
    
    for batch in dataloader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        with torch.no_grad():
            losses = policy.compute_loss(batch)
            all_metrics.append({
                'loss': losses['loss'].item(),
                'v_loss': losses['v_loss'].item(),
                'q_loss': losses['q_loss'].item(),
            })
    
    # Average metrics
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics])
        for k in all_metrics[0].keys()
    }
    
    return avg_metrics


def main():
    args = parse_args()
    
    # Initialize distributed
    distributed_manager = DistributedManager()
    distributed_manager.initialize()
    
    # Load config
    import yaml
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    
    config = PiStarRECAPConfig.from_dict(config_dict)
    
    # Override config with args
    config.training.batch_size = args.batch_size
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    if distributed_manager.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Logging
    log_dir = setup_logging(args, config, distributed_manager)
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        args.dataset_path,
        args.batch_size,
        args.num_workers,
        distributed_manager,
    )
    
    # Get dataset stats for normalization
    dataset = train_dataloader.dataset
    if hasattr(dataset, 'stats'):
        dataset_stats = dataset.stats
    else:
        dataset_stats = None
    
    # Create policy
    policy = PiStarRECAPPolicy(config, dataset_stats=dataset_stats)
    policy = policy.cuda()
    
    # Setup FSDP if enabled
    if args.use_fsdp and distributed_manager.world_size > 1:
        from lerobot.policies.pi_star_recap.distributed import setup_fsdp
        policy = setup_fsdp(
            policy,
            distributed_manager.local_rank,
            mixed_precision=config.training.amp_dtype,
            strategy=config.distributed.fsdp_strategy,
        )
    
    # Setup optimizer
    policy.configure_optimizers()
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        policy.load_checkpoint(args.resume)
        start_epoch = policy.epoch
        if distributed_manager.is_main_process():
            print(f"Resumed from checkpoint: epoch {start_epoch}, step {policy.global_step}")
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        if distributed_manager.is_main_process():
            print(f"\nEpoch {epoch}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch(
            policy,
            train_dataloader,
            epoch,
            args,
            distributed_manager,
        )
        
        if distributed_manager.is_main_process():
            print(f"Epoch {epoch} finished: {train_metrics}")
            
            # Save epoch checkpoint
            epoch_checkpoint = output_dir / f"checkpoint_epoch_{epoch}.pt"
            policy.save_checkpoint(
                str(epoch_checkpoint),
                metadata={'epoch': epoch, 'metrics': train_metrics}
            )
        
        # Sync before next epoch
        distributed_manager.barrier()
    
    # Cleanup
    distributed_manager.cleanup()
    
    if distributed_manager.is_main_process():
        print("Training complete!")


if __name__ == "__main__":
    main()
