#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


@dataclass
class ValidationConfig:
    """Configuration for validation during training."""
    enable: bool = True
    val_ratio: float = 0.2  # Fraction of episodes to use for validation
    val_freq: int = 500     # How often to run validation (in training steps)
    val_batch_size: int = 8 # Batch size for validation
    save_split: bool = True # Whether to save the train/val split to disk
    log_train_eval_loss: bool = False  # Whether to also log training loss in eval mode
    train_eval_freq: int = 1000  # How often to compute training loss in eval mode (in steps)


@dataclass
class TrainValPipelineConfig(TrainPipelineConfig):
    """Extended training configuration with validation support."""
    validation: ValidationConfig = field(default_factory=ValidationConfig)


def create_train_val_split(
    dataset_repo_id: str,
    val_ratio: float = 0.2,
    output_dir: Path = None,
    save_split: bool = True,
    seed: int = 1000
) -> tuple[list[int], list[int]]:
    """
    Create train/validation split and optionally save to disk.
    
    Args:
        dataset_repo_id: Repository ID of the dataset
        val_ratio: Fraction of episodes to use for validation
        output_dir: Directory to save split configuration
        save_split: Whether to save split to disk
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_episodes, val_episodes)
    """
    # Check if split already exists
    split_file = output_dir / "train_val_split.json" if output_dir else None
    
    if split_file and split_file.exists():
        logging.info(f"Loading existing train/val split from {split_file}")
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        return split_data["train_episodes"], split_data["val_episodes"]
    
    # Create new split
    logging.info(f"Creating new train/val split for {dataset_repo_id}")
    meta = LeRobotDatasetMetadata(dataset_repo_id)
    total_episodes = meta.total_episodes
    
    logging.info(f"Total episodes in dataset: {total_episodes}")
    
    # Create episode indices
    all_episodes = list(range(total_episodes))
    
    # Shuffle episodes for random split (but deterministic with seed)
    if seed is not None:
        torch.manual_seed(seed)
        indices = torch.randperm(total_episodes).tolist()
        all_episodes = [all_episodes[i] for i in indices]
    
    # Calculate split point
    num_val = math.ceil(total_episodes * val_ratio)
    num_train = total_episodes - num_val
    
    # Split episodes
    train_episodes = sorted(all_episodes[:num_train])
    val_episodes = sorted(all_episodes[num_train:])
    
    logging.info(f"Train episodes: {len(train_episodes)} ({len(train_episodes)/total_episodes*100:.1f}%)")
    logging.info(f"Val episodes: {len(val_episodes)} ({len(val_episodes)/total_episodes*100:.1f}%)")
    logging.info(f"Train episodes: {train_episodes[:10]}{'...' if len(train_episodes) > 10 else ''}")
    logging.info(f"Val episodes: {val_episodes[:10]}{'...' if len(val_episodes) > 10 else ''}")
    
    # Check for overlap
    overlap = set(train_episodes) & set(val_episodes)
    if overlap:
        logging.error(f"OVERLAP DETECTED! Episodes {overlap} appear in both train and validation sets!")
    else:
        logging.info("✓ No overlap between train and validation episodes")
    
    # Save split to disk
    if save_split and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        split_data = {
            "dataset_repo_id": dataset_repo_id,
            "total_episodes": total_episodes,
            "val_ratio": val_ratio,
            "seed": seed,
            "train_episodes": train_episodes,
            "val_episodes": val_episodes,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        logging.info(f"Saved train/val split to {split_file}")
    
    return train_episodes, val_episodes


def create_validation_dataset(cfg: TrainValPipelineConfig, val_episodes: list[int]) -> LeRobotDataset:
    """Create validation dataset with specified episodes."""
    # Create a copy of dataset config with validation episodes
    val_dataset_config = DatasetConfig(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=val_episodes,
        image_transforms=cfg.dataset.image_transforms,
        revision=cfg.dataset.revision,
        use_imagenet_stats=cfg.dataset.use_imagenet_stats,
        video_backend=cfg.dataset.video_backend
    )
    
    # Create temporary config for validation dataset
    val_cfg = TrainValPipelineConfig(
        dataset=val_dataset_config,
        policy=cfg.policy
    )
    
    dataset = make_dataset(val_cfg)
    
    # Fix episode indexing issue by creating a custom episode mapping
    # The issue is that episode_data_index is indexed by position in filtered list,
    # but the dataset items still have original episode indices
    
    logging.info(f"Applying episode index mapping for validation dataset with {len(val_episodes)} episodes")
    # Create mapping from original episode index to position in filtered list
    original_to_filtered_idx = {ep_idx: i for i, ep_idx in enumerate(val_episodes)}
    
    # Monkey patch the _get_query_indices method to handle the mapping correctly
    original_get_query_indices = dataset._get_query_indices
    
    def fixed_get_query_indices(idx: int, ep_idx: int):
        # Map original episode index to filtered position
        if ep_idx in original_to_filtered_idx:
            filtered_ep_idx = original_to_filtered_idx[ep_idx]
            return original_get_query_indices(idx, filtered_ep_idx)
        else:
            # Should not happen if episodes are filtered correctly
            raise ValueError(f"Episode {ep_idx} not found in validation episodes {val_episodes}")
    
    dataset._get_query_indices = fixed_get_query_indices
    return dataset


def run_validation(
    policy: PreTrainedPolicy,
    val_dataset: LeRobotDataset,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 4
) -> dict[str, float]:
    """Run validation and return metrics."""
    policy.eval()
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation - preserve episode structure for ACT
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    
    total_loss = 0.0
    total_samples = 0
    loss_dict_sum = {}
    
    with torch.no_grad():
        for batch in val_dataloader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            
            # Forward pass
            loss, loss_dict = policy.forward(batch)
            
            batch_size_actual = batch["index"].shape[0]
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual
            
            # Accumulate loss components
            if loss_dict:
                for key, value in loss_dict.items():
                    if key not in loss_dict_sum:
                        loss_dict_sum[key] = 0.0
                    loss_dict_sum[key] += value * batch_size_actual
    
    # Calculate average losses
    avg_loss = total_loss / total_samples
    avg_loss_dict = {key: value / total_samples for key, value in loss_dict_sum.items()}
    
    return {
        "val_loss": avg_loss,
        **{f"val_{key}": value for key, value in avg_loss_dict.items()}
    }


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainValPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Create train/validation split
    if cfg.validation.enable:
        train_episodes, val_episodes = create_train_val_split(
            dataset_repo_id=cfg.dataset.repo_id,
            val_ratio=cfg.validation.val_ratio,
            output_dir=cfg.output_dir,
            save_split=cfg.validation.save_split,
            seed=cfg.seed
        )
        
        # Update dataset config to use only training episodes
        cfg.dataset.episodes = train_episodes
        
        logging.info("Creating validation dataset")
        val_dataset = create_validation_dataset(cfg, val_episodes)
    else:
        val_dataset = None

    logging.info("Creating training dataset")
    dataset = make_dataset(cfg)
    
    # Fix episode indexing issue for training dataset as well
    if cfg.validation.enable and train_episodes:
        logging.info(f"Applying episode index mapping for training dataset with {len(train_episodes)} episodes")
        # Create mapping from original episode index to position in filtered list
        original_to_filtered_idx = {ep_idx: i for i, ep_idx in enumerate(train_episodes)}
        
        # Monkey patch the _get_query_indices method to handle the mapping correctly
        original_get_query_indices = dataset._get_query_indices
        
        def fixed_get_query_indices(idx: int, ep_idx: int):
            # Map original episode index to filtered position
            if ep_idx in original_to_filtered_idx:
                filtered_ep_idx = original_to_filtered_idx[ep_idx]
                return original_get_query_indices(idx, filtered_ep_idx)
            else:
                # Should not happen if episodes are filtered correctly
                raise ValueError(f"Episode {ep_idx} not found in training episodes {train_episodes}")
        
        # Only apply the fix if the dataset uses delta_indices (temporal context)
        if hasattr(dataset, 'delta_indices') and dataset.delta_indices is not None:
            dataset._get_query_indices = fixed_get_query_indices

    # Create environment used for evaluating checkpoints during training on simulation data.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"Train {dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"Train {dataset.num_episodes=}")
    if val_dataset:
        logging.info(f"Val frames: {val_dataset.num_frames} ({format_big_number(val_dataset.num_frames)})")
        logging.info(f"Val episodes: {val_dataset.num_episodes}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    # For ACT policy, use episode-aware sampling to respect temporal structure
    if cfg.policy.type == "act" or hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
            shuffle=True,  # Shuffle within episode bounds
        )
        logging.info("Using EpisodeAwareSampler for ACT policy to preserve temporal structure")
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_env_eval_step = cfg.env and cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        is_val_step = (cfg.validation.enable and val_dataset and 
                      cfg.validation.val_freq > 0 and step % cfg.validation.val_freq == 0)
        is_train_eval_step = (cfg.validation.log_train_eval_loss and 
                             cfg.validation.train_eval_freq > 0 and 
                             step % cfg.validation.train_eval_freq == 0)

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        # Run validation
        if is_val_step or is_train_eval_step:
            metrics_to_log = {}
            
            # Run validation if scheduled
            if is_val_step:
                logging.info(f"Running validation at step {step}")
                start_val_time = time.perf_counter()
                
                val_metrics = run_validation(
                    policy=policy,
                    val_dataset=val_dataset,
                    device=device,
                    batch_size=cfg.validation.val_batch_size,
                    num_workers=cfg.num_workers
                )
                
                val_time = time.perf_counter() - start_val_time
                val_metrics["val_time_s"] = val_time
                metrics_to_log.update(val_metrics)
                logging.info(f"Validation metrics: {val_metrics}")
            
            # Compute training loss in eval mode if scheduled
            if is_train_eval_step:
                logging.info(f"Computing training loss in eval mode at step {step}")
                start_train_eval_time = time.perf_counter()
                
                train_eval_metrics = run_validation(
                    policy=policy,
                    val_dataset=dataset,  # Use training dataset
                    device=device,
                    batch_size=cfg.validation.val_batch_size,
                    num_workers=cfg.num_workers
                )
                
                train_eval_time = time.perf_counter() - start_train_eval_time
                
                # Rename metrics to distinguish from validation
                train_eval_metrics = {k.replace("val_", "train_eval_"): v for k, v in train_eval_metrics.items()}
                train_eval_metrics["train_eval_time_s"] = train_eval_time
                metrics_to_log.update(train_eval_metrics)
                logging.info(f"Training eval metrics: {train_eval_metrics}")
                
                # Log the ratio to show dropout effect if both are available
                if "val_loss" in metrics_to_log and "train_eval_loss" in metrics_to_log:
                    metrics_to_log["dropout_effect_ratio"] = metrics_to_log["val_loss"] / metrics_to_log["train_eval_loss"]
            
            # Log all metrics to wandb
            if wandb_logger and metrics_to_log:
                wandb_logger.log_dict(metrics_to_log, step, mode="eval")

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        # Environment evaluation (if provided)
        if is_env_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()

# Example usage:
# python train_val.py \
#     --policy.type=act \
#     --dataset.repo_id=jackvial/merged_datasets_test_2 \
#     --output_dir=outputs/train/act_koch_screwdriver_with_validation \
#     --steps=10000 \
#     --log_freq=100 \
#     --validation.val_freq=200 \
#     --validation.enable=true \
#     --validation.val_ratio=0.2 \
#     --validation.log_train_eval_loss=true \
#     --validation.train_eval_freq=1000 \
#     --batch_size=8 \
#     --wandb.enable=true \
#     --wandb.project=lerobot_training
#
# Key features:
# - Uses EpisodeAwareSampler to preserve temporal structure in action chunks
# - Proper train/validation split with no data leakage
# - Optional train_eval_loss computation to see dropout effect
# - Separate frequencies for validation (val_freq) and train eval loss (train_eval_freq) 