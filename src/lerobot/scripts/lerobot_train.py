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
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any
import json

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)


class NormalizationRangeTracker:
    """
    Tracks the maximum range encountered for normalized data to verify normalization stays within [-1, 1].
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked statistics."""
        self.max_ranges = {}
    
    def update_and_get_ranges(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        Update and return the maximum ranges encountered for normalized data.
        
        Args:
            batch: Dictionary containing normalized tensors
        
        Returns:
            Dictionary with maximum range values for each feature to verify normalization
        """
        ranges = {}
        
        # Track state data (observations)
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and key.startswith("observation.state"):
                # Remove batch dimension for range calculation if present
                data = tensor.view(-1) if tensor.numel() > 0 else tensor
                if data.numel() > 0:
                    current_min = float(data.min().item())
                    current_max = float(data.max().item())
                    current_range = current_max - current_min
                    
                    # Update maximum range encountered
                    feature_key = key.replace("observation.", "")  # Clean up key name
                    if feature_key not in self.max_ranges:
                        self.max_ranges[feature_key] = {
                            "min": current_min,
                            "max": current_max,
                            "range": current_range
                        }
                    else:
                        # Expand the overall range if we see new extremes
                        old_min = self.max_ranges[feature_key]["min"]
                        old_max = self.max_ranges[feature_key]["max"]
                        new_min = min(old_min, current_min)
                        new_max = max(old_max, current_max)
                        self.max_ranges[feature_key] = {
                            "min": new_min,
                            "max": new_max,
                            "range": new_max - new_min
                        }
                    
                    # Log the maximum range encountered so far
                    ranges[f"normalized_{feature_key}_max_range"] = self.max_ranges[feature_key]["range"]
        
        # Track action data
        if "action" in batch and isinstance(batch["action"], torch.Tensor):
            action_tensor = batch["action"]
            # Remove batch dimension for range calculation if present
            data = action_tensor.view(-1) if action_tensor.numel() > 0 else action_tensor
            if data.numel() > 0:
                current_min = float(data.min().item())
                current_max = float(data.max().item())
                current_range = current_max - current_min
                
                # Update maximum range encountered
                if "action" not in self.max_ranges:
                    self.max_ranges["action"] = {
                        "min": current_min,
                        "max": current_max,
                        "range": current_range
                    }
                else:
                    # Expand the overall range if we see new extremes
                    old_min = self.max_ranges["action"]["min"]
                    old_max = self.max_ranges["action"]["max"]
                    new_min = min(old_min, current_min)
                    new_max = max(old_max, current_max)
                    self.max_ranges["action"] = {
                        "min": new_min,
                        "max": new_max,
                        "range": new_max - new_min
                    }
                
                # Log the maximum range encountered so far
                ranges["normalized_action_max_range"] = self.max_ranges["action"]["range"]
        
        return ranges


def save_normalized_images(batch: dict[str, Any], step: int, output_dir: Path, wandb_logger=None) -> list[Path]:
    """
    Save normalized images from batch and optionally push to wandb as artifacts.
    
    Args:
        batch: Dictionary containing normalized tensors including images
        step: Current training step
        output_dir: Directory to save images
        wandb_logger: Optional wandb logger for artifact pushing
        
    Returns:
        List of saved image paths
    """
    saved_paths = []
    images_dir = output_dir / "normalized_images" / f"step_{step}"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for image keys in the batch
    for key, tensor in batch.items():
        if isinstance(tensor, torch.Tensor) and "image" in key.lower():
            # Assume images are in format [batch_size, channels, height, width]
            if len(tensor.shape) == 4 and tensor.shape[1] in [1, 3]:  # grayscale or RGB
                for batch_idx in range(tensor.shape[0]):
                    img_tensor = tensor[batch_idx]
                    
                    # Convert from normalized [-1, 1] to [0, 1] range
                    img_tensor = (img_tensor + 1.0) / 2.0
                    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
                    
                    # Convert to PIL Image
                    to_pil = transforms.ToPILImage()
                    img = to_pil(img_tensor)
                    
                    # Save image
                    clean_key = key.replace(".", "_").replace("/", "_")
                    img_path = images_dir / f"{clean_key}_batch_{batch_idx}.png"
                    img.save(img_path)
                    saved_paths.append(img_path)
                    
                logging.info(f"Saved {len(tensor)} images for key '{key}' at step {step}")
    
    # Push to wandb as artifact if logger is available
    if wandb_logger and saved_paths:
        try:
            import wandb
            artifact = wandb.Artifact(
                name=f"normalized_images_step_{step}",
                type="images",
                description=f"Normalized input images at training step {step}"
            )
            for img_path in saved_paths:
                artifact.add_file(str(img_path))
            wandb_logger.wandb_run.log_artifact(artifact)
            logging.info(f"Pushed {len(saved_paths)} normalized images to wandb for step {step}")
        except Exception as e:
            logging.warning(f"Failed to push images to wandb: {e}")
    
    return saved_paths


def save_normalized_state_action_data(batch: dict[str, Any], step: int, output_dir: Path, wandb_logger=None) -> Path:
    """
    Save normalized observation.state and action data to a JSON file and optionally push to wandb.
    
    Args:
        batch: Dictionary containing normalized tensors
        step: Current training step
        output_dir: Directory to save data
        wandb_logger: Optional wandb logger for artifact pushing
        
    Returns:
        Path to saved data file
    """
    data_dir = output_dir / "normalized_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    data_file = data_dir / f"step_{step}_state_action.json"
    
    # Collect state and action data
    step_data = {
        "step": step,
        "observation_state": {},
        "action": None
    }
    
    # Extract observation.state data
    for key, tensor in batch.items():
        if isinstance(tensor, torch.Tensor) and key.startswith("observation.state"):
            # Convert tensor to numpy and then to list for JSON serialization
            numpy_data = tensor.detach().cpu().numpy()
            step_data["observation_state"][key] = {
                "shape": list(numpy_data.shape),
                "data": numpy_data.tolist(),
                "min": float(numpy_data.min()),
                "max": float(numpy_data.max()),
                "mean": float(numpy_data.mean()),
                "std": float(numpy_data.std())
            }
    
    # Extract action data
    if "action" in batch and isinstance(batch["action"], torch.Tensor):
        action_tensor = batch["action"]
        numpy_data = action_tensor.detach().cpu().numpy()
        step_data["action"] = {
            "shape": list(numpy_data.shape),
            "data": numpy_data.tolist(),
            "min": float(numpy_data.min()),
            "max": float(numpy_data.max()),
            "mean": float(numpy_data.mean()),
            "std": float(numpy_data.std())
        }
    
    # Save to JSON file
    with open(data_file, 'w') as f:
        json.dump(step_data, f, indent=2)
    
    logging.info(f"Saved normalized state/action data for step {step} to {data_file}")
    
    # Push to wandb as artifact if logger is available
    if wandb_logger:
        try:
            import wandb
            artifact = wandb.Artifact(
                name=f"normalized_state_action_step_{step}",
                type="data",
                description=f"Normalized observation.state and action data at training step {step}"
            )
            artifact.add_file(str(data_file))
            wandb_logger.wandb_run.log_artifact(artifact)
            logging.info(f"Pushed normalized state/action data to wandb for step {step}")
        except Exception as e:
            logging.warning(f"Failed to push state/action data to wandb: {e}")
    
    return data_file


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. It also handles mixed-precision training via a GradScaler.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        grad_scaler: The GradScaler for automatic mixed-precision training.
        lr_scheduler: An optional learning rate scheduler.
        use_amp: A boolean indicating whether to use automatic mixed precision.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.tensor(0.0, device=accelerator.device)

    # Use accelerator's optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

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
def train(cfg: TrainPipelineConfig):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
    """
    cfg.validate()

    # Initialize accelerator with DDP configuration for unused parameters
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=getattr(cfg, "gradient_accumulation_steps", 1),
        mixed_precision=getattr(cfg, "mixed_precision", "no"),
        log_with="wandb" if cfg.wandb.enable and cfg.wandb.project else None,
        project_dir=str(cfg.output_dir) if cfg.wandb.enable and cfg.wandb.project else None,
        kwargs_handlers=[ddp_kwargs],
    )

    # Only log on main process
    if accelerator.is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and accelerator.is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if accelerator.is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Use accelerator's device instead of manual device selection
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if accelerator.is_main_process:
        logging.info("Creating dataset")
        # Main process downloads/loads the dataset first
        dataset = make_dataset(cfg)

    # Wait for main process to finish downloading dataset
    accelerator.wait_for_everyone()

    # Now all processes can safely load the dataset
    if not accelerator.is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if accelerator.is_main_process:
            logging.info("Creating env")
            eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
        else:
            eval_env = None

    if accelerator.is_main_process:
        logging.info("Creating policy")

    # All processes create the policy, but we ensure dataset metadata is available
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logging.info("Creating optimizer and scheduler")

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats, 
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats, 
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    # Scale learning rate and scheduler parameters for distributed training
    original_lr = cfg.optimizer.lr
    scaled_lr = original_lr * accelerator.num_processes
    cfg.optimizer.lr = scaled_lr

    # Scale scheduler parameters to account for faster data consumption in distributed training
    original_scheduler_params = {}
    if cfg.scheduler is not None:
        # Store original values
        if hasattr(cfg.scheduler, "num_warmup_steps"):
            original_scheduler_params["num_warmup_steps"] = cfg.scheduler.num_warmup_steps
            cfg.scheduler.num_warmup_steps = int(cfg.scheduler.num_warmup_steps * accelerator.num_processes)

        if hasattr(cfg.scheduler, "num_decay_steps"):
            original_scheduler_params["num_decay_steps"] = cfg.scheduler.num_decay_steps
            cfg.scheduler.num_decay_steps = int(cfg.scheduler.num_decay_steps * accelerator.num_processes)

    if accelerator.is_main_process:
        logging.info(
            f"Scaling learning rate from {original_lr} to {scaled_lr} for {accelerator.num_processes} processes"
        )
        if cfg.scheduler is not None and original_scheduler_params:
            for param_name, original_value in original_scheduler_params.items():
                new_value = getattr(cfg.scheduler, param_name)
                logging.info(f"Scaling scheduler {param_name} from {original_value} to {new_value}")

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # No need for GradScaler when using accelerator's mixed precision
    _grad_scaler = GradScaler(device.type, enabled=False)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if accelerator.is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(
            f"Effective batch size: {cfg.batch_size} x {accelerator.num_processes} = {cfg.batch_size * accelerator.num_processes}"
        )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2,
    )

    # Check for normalization issues and warn about missing features (BEFORE accelerator wrapping)
    if accelerator.is_main_process:
        logging.info("=== NORMALIZATION VALIDATION ===")
        
        # Get all expected features from policy config
        all_policy_features = {**policy.config.input_features, **policy.config.output_features}
        logging.info(f"Policy expects {len(all_policy_features)} features: {list(all_policy_features.keys())}")
        
        # Get normalizer step info
        normalizer_step = None
        for step_obj in preprocessor.steps:
            if hasattr(step_obj, 'norm_map') and hasattr(step_obj, 'features'):
                normalizer_step = step_obj
                break
        
        if normalizer_step is not None:
            normalizer_features = set(normalizer_step.features.keys())
            available_stats = set(normalizer_step.stats.keys()) if normalizer_step.stats else set()
            
            # Check for policy features not in normalizer
            missing_from_normalizer = set(all_policy_features.keys()) - normalizer_features
            if missing_from_normalizer:
                logging.warning(f"⚠️  {len(missing_from_normalizer)} policy features NOT in normalizer: {sorted(missing_from_normalizer)}")
            
            # Check for normalizer features without stats
            missing_stats = normalizer_features - available_stats
            if missing_stats:
                logging.warning(f"⚠️  {len(missing_stats)} normalizer features MISSING stats: {sorted(missing_stats)}")
            
            # Check for features requiring QUANTILES but missing q01/q99
            for feature_name in normalizer_features & available_stats:
                feature_type = normalizer_step.features[feature_name].type
                if feature_type in normalizer_step.norm_map:
                    norm_mode = normalizer_step.norm_map[feature_type]
                    if norm_mode.name == "QUANTILES":
                        stats = normalizer_step.stats[feature_name]
                        if 'q01' not in stats or 'q99' not in stats:
                            logging.warning(f"⚠️  Feature '{feature_name}' uses QUANTILES but missing q01/q99 stats!")
            
            # Summary of what will actually be normalized
            will_be_normalized = normalizer_features & available_stats
            logging.info(f"✅ {len(will_be_normalized)} features WILL be normalized: {sorted(will_be_normalized)}")
            
            if missing_from_normalizer or missing_stats:
                total_issues = len(missing_from_normalizer) + len(missing_stats)
                logging.warning(f"🚨 {total_issues} features will NOT be normalized due to config/stats mismatches!")
                logging.warning("   This will cause data to remain in original ranges instead of [-1, 1]")
        else:
            logging.warning("🚨 No normalizer step found in preprocessor! Data will not be normalized!")
        
        logging.info("=== END NORMALIZATION VALIDATION ===")

    # Ensure all processes are synchronized before preparing with accelerator
    accelerator.wait_for_everyone()

    # Prepare model, optimizer, scheduler, and dataloader with accelerator
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    if lr_scheduler is not None:
        lr_scheduler = accelerator.prepare(lr_scheduler)

    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    # Initialize normalization range tracker
    normalization_tracker = NormalizationRangeTracker()

    if accelerator.is_main_process:
        logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        
        # Track maximum ranges for normalized data to verify normalization is working
        normalization_ranges = normalization_tracker.update_and_get_ranges(batch)
        
        # Save normalized data for first 10 steps (only on main process to avoid duplicates)
        if step < 10 and accelerator.is_main_process:
            try:
                # Save normalized images
                save_normalized_images(batch, step, cfg.output_dir, wandb_logger)
                
                # Save normalized state/action data  
                save_normalized_state_action_data(batch, step, cfg.output_dir, wandb_logger)
                
            except Exception as e:
                logging.warning(f"Failed to save normalized data for step {step}: {e}")
        
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step and accelerator.is_main_process:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                
                # Add normalization range logging to verify normalization is working
                wandb_log_dict.update(normalization_ranges)
                
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if accelerator.is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)

                # Use accelerator to unwrap the model before saving
                unwrapped_policy = accelerator.unwrap_model(policy)
                save_checkpoint(
                    checkpoint_dir,
                    step,
                    cfg,
                    unwrapped_policy,
                    optimizer,
                    lr_scheduler,
                    preprocessor,
                    postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            # Wait for all processes after checkpoint is saved
            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            # Only evaluate on main process
            if accelerator.is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")

                # Use unwrapped model for evaluation and accelerator's autocast
                unwrapped_policy = accelerator.unwrap_model(policy)
                with (
                    torch.no_grad(),
                    accelerator.autocast(),
                ):
                    eval_info = eval_policy_all(
                        envs=eval_env,
                        policy=unwrapped_policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks if hasattr(cfg.env, 'max_parallel_tasks') else None,
                    )

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
                )
                # Extract overall aggregated metrics
                aggregated = eval_info["overall"]
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                logging.info(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            # Wait for evaluation to complete before continuing training
            accelerator.wait_for_everyone()

    if eval_env:
        eval_env.close()

    if accelerator.is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            # Use unwrapped model for pushing to hub
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
