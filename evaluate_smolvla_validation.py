#!/usr/bin/env python

"""
Validation loss evaluation for SmolVLA models.
Adapted from lerobot/examples/advanced/2_calculate_validation_loss.py
"""

import math
import logging
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_logging, get_safe_torch_device
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs import parser


def calculate_validation_metrics(
    policy,
    val_dataloader,
    device: torch.device,
    num_action_dims: int = 6
) -> dict:
    """Calculate validation metrics including loss and action prediction accuracy."""
    
    policy.eval()
    
    loss_cumsum = 0
    action_l2_cumsum = 0
    action_l1_cumsum = 0
    n_examples_evaluated = 0
    
    logging.info(f"Running validation on {len(val_dataloader)} batches...")
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Forward pass
            loss, output_dict = policy.forward(batch)
            
            # Accumulate loss
            loss_cumsum += loss.item()
            batch_size = batch["action"].shape[0]
            n_examples_evaluated += batch_size
            
            # Calculate action prediction accuracy if we have predictions
            if output_dict and "action" in output_dict:
                pred_actions = output_dict["action"]
                true_actions = batch["action"]
                
                # L2 distance (RMSE)
                l2_dist = torch.norm(pred_actions - true_actions, dim=-1)
                action_l2_cumsum += l2_dist.sum().item()
                
                # L1 distance (MAE)
                l1_dist = torch.abs(pred_actions - true_actions).mean(dim=-1)
                action_l1_cumsum += l1_dist.sum().item()
    
    # Calculate averages
    avg_loss = loss_cumsum / n_examples_evaluated
    avg_action_l2 = action_l2_cumsum / n_examples_evaluated
    avg_action_l1 = action_l1_cumsum / n_examples_evaluated
    
    metrics = {
        "validation_loss": avg_loss,
        "action_rmse": avg_action_l2,
        "action_mae": avg_action_l1,
        "n_examples": n_examples_evaluated
    }
    
    return metrics


@parser.wrap()
def main(cfg: TrainPipelineConfig):
    """Main validation evaluation function."""
    
    init_logging()
    
    # Check device
    device = get_safe_torch_device(cfg.policy.device, log=True)
    
    logging.info(f"Loading policy from: {cfg.policy.path}")
    
    # Load datasets - use the same repo_ids as training
    if cfg.dataset.repo_id == "all/datasets":
        dataset_repo_ids = [
            "danielkorth/whiteboard-marker", 
            "danielkorth/usbc-cable-2", 
            "danielkorth/bike-light", 
            "danielkorth/usb-stick"
        ]
    else:
        dataset_repo_ids = [cfg.dataset.repo_id] if isinstance(cfg.dataset.repo_id, str) else cfg.dataset.repo_id
    
    all_val_metrics = {}
    
    for repo_id in dataset_repo_ids:
        logging.info(f"Evaluating on dataset: {repo_id}")
        
        # Load dataset metadata
        dataset_metadata = LeRobotDatasetMetadata(repo_id)
        total_episodes = dataset_metadata.total_episodes
        
        if total_episodes < 2:
            logging.warning(f"Dataset {repo_id} has only {total_episodes} episodes, skipping validation split")
            continue
        
        # Create train/val split (80/20)
        episodes = list(range(total_episodes))
        num_train_episodes = math.floor(total_episodes * 0.8)
        val_episodes = episodes[num_train_episodes:]
        
        logging.info(f"Total episodes: {total_episodes}, Validation episodes: {len(val_episodes)}")
        
        if len(val_episodes) == 0:
            logging.warning(f"No validation episodes for {repo_id}, skipping")
            continue
        
        # Load validation dataset
        val_dataset = LeRobotDataset(
            repo_id=repo_id,
            episodes=val_episodes,
            video_backend=cfg.dataset.video_backend
        )
        
        # Load policy (need dataset metadata)
        policy = make_policy(cfg=cfg.policy, ds_meta=val_dataset.meta)
        policy.eval()
        policy.to(device)
        
        # Create validation dataloader
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=4,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=device.type != "cpu",
            drop_last=False,
        )
        
        # Calculate validation metrics
        val_metrics = calculate_validation_metrics(
            policy=policy,
            val_dataloader=val_dataloader,
            device=device,
            num_action_dims=val_dataset.meta.action_space.shape[0]
        )
        
        all_val_metrics[repo_id] = val_metrics
        
        # Print results for this dataset
        logging.info(f"Results for {repo_id}:")
        for metric_name, value in val_metrics.items():
            if metric_name != "n_examples":
                logging.info(f"  {metric_name}: {value:.6f}")
        
        # Clean up
        del policy
        del val_dataset
        del val_dataloader
        torch.cuda.empty_cache()
    
    # Print overall summary
    logging.info("\n" + "="*50)
    logging.info("VALIDATION SUMMARY")
    logging.info("="*50)
    
    for repo_id, metrics in all_val_metrics.items():
        logging.info(f"\n{repo_id}:")
        logging.info(f"  Validation Loss: {metrics['validation_loss']:.6f}")
        logging.info(f"  Action RMSE: {metrics['action_rmse']:.6f}")
        logging.info(f"  Action MAE: {metrics['action_mae']:.6f}")
        logging.info(f"  Examples: {metrics['n_examples']}")
    
    # Calculate overall averages weighted by number of examples
    if all_val_metrics:
        total_examples = sum(m["n_examples"] for m in all_val_metrics.values())
        avg_loss = sum(m["validation_loss"] * m["n_examples"] for m in all_val_metrics.values()) / total_examples
        avg_rmse = sum(m["action_rmse"] * m["n_examples"] for m in all_val_metrics.values()) / total_examples
        avg_mae = sum(m["action_mae"] * m["n_examples"] for m in all_val_metrics.values()) / total_examples
        
        logging.info(f"\nOVERALL WEIGHTED AVERAGES:")
        logging.info(f"  Validation Loss: {avg_loss:.6f}")
        logging.info(f"  Action RMSE: {avg_rmse:.6f}")
        logging.info(f"  Action MAE: {avg_mae:.6f}")
        logging.info(f"  Total Examples: {total_examples}")


if __name__ == "__main__":
    main() 