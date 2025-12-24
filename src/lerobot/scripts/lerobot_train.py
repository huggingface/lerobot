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
import shutil
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
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


class EarlyStoppingTracker:
    """Tracks validation metrics and determines when to stop training early."""

    def __init__(
        self,
        patience_steps: int,
        min_delta: float = 0.0,
        higher_is_better: bool = False,
    ):
        self.patience_steps = patience_steps
        self.min_delta = min_delta
        self.higher_is_better = higher_is_better
        self.best_value: float | None = None
        self.best_step: int = 0
        self.steps_without_improvement: int = 0

    def update(self, value: float, step: int) -> bool:
        """Update tracker with new metric value.

        Returns True if training should stop (patience exceeded).
        """
        if self.best_value is None:
            self.best_value = value
            self.best_step = step
            return False

        # Check if this is an improvement
        if self.higher_is_better:
            improved = value > self.best_value * (1 + self.min_delta)
        else:
            improved = value < self.best_value * (1 - self.min_delta)

        if improved:
            self.best_value = value
            self.best_step = step
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement = step - self.best_step

        return self.steps_without_improvement >= self.patience_steps

    def get_status(self) -> dict:
        """Get current early stopping status for logging."""
        return {
            "best_value": self.best_value,
            "best_step": self.best_step,
            "steps_without_improvement": self.steps_without_improvement,
        }


def cleanup_old_checkpoints(output_dir, keep_last_n: int, current_step: int) -> None:
    """Remove old checkpoints, keeping only the last N plus the 'last' symlink target."""
    if keep_last_n <= 0:
        return

    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return

    # Find all step checkpoint directories (format: NNNNNN or step_NNNNNN)
    checkpoint_dirs = []
    for d in checkpoints_dir.iterdir():
        if d.is_dir() and d.name != "last":
            # Try to extract step number from directory name
            try:
                # Handle formats like "005000" or "step_005000"
                name = d.name.replace("step_", "")
                step_num = int(name)
                checkpoint_dirs.append((step_num, d))
            except ValueError:
                continue

    # Sort by step number (oldest first)
    checkpoint_dirs.sort(key=lambda x: x[0])

    # Keep the last N checkpoints
    if len(checkpoint_dirs) > keep_last_n:
        to_remove = checkpoint_dirs[:-keep_last_n]
        last_symlink = checkpoints_dir / "last"
        last_resolved = last_symlink.resolve() if last_symlink.exists() else None

        for step_num, checkpoint_dir in to_remove:
            # Never remove the checkpoint pointed to by the 'last' symlink
            if last_resolved and checkpoint_dir.resolve() == last_resolved:
                logging.info(f"Skipping removal of current 'last' checkpoint: {checkpoint_dir}")
                continue

            logging.info(f"Removing old checkpoint: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        rabc_weights_provider: Optional RABCWeights instance for sample weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        # Use per-sample loss when RA-BC is enabled for proper weighting
        if rabc_batch_weights is not None:
            # Get per-sample losses
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Apply RA-BC weights: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
            # rabc_batch_weights is already normalized to sum to batch_size
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            # Log raw mean weight (before normalization) - this is the meaningful metric
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def validate_dataset_loss(
    policy: PreTrainedPolicy,
    dataloader: torch.utils.data.DataLoader,
    preprocessor,
    accelerator: Accelerator,
) -> dict[str, float]:
    """Validate using actual inference (select_action) for model-agnostic L1/L2 loss.

    This mimics real inference by calling select_action and comparing predicted
    actions with ground truth, providing metrics that are comparable across
    different policy architectures.

    Note: The dataloader should have batch_size=1 because select_action in many
    policies (e.g. ACT, SmolVLA) uses internal temporal queues that are not
    compatible with batching during inference.
    """
    from lerobot.utils.constants import ACTION

    policy.eval()

    l1_acc = 0.0
    l2_acc = 0.0
    count = 0

    # Unwrap policy from accelerator wrapper to access reset/select_action methods
    unwrapped_policy = accelerator.unwrap_model(policy)

    with torch.no_grad(), accelerator.autocast():
        for batch in dataloader:
            batch = preprocessor(batch)

            # Extract ground truth actions before select_action potentially pops them
            gt_actions = batch[ACTION].clone()

            # Reset policy to clear any internal queues/state
            unwrapped_policy.reset()

            # Run actual inference - this is the model-agnostic path
            # select_action returns shape (batch_size, action_dim)
            pred_actions = unwrapped_policy.select_action(batch)

            # Handle different action shapes:
            # - gt_actions may be (B, action_dim) or (B, horizon, action_dim)
            # - pred_actions is typically (B, action_dim) for single step
            if gt_actions.dim() == 3:
                # Ground truth has temporal dimension, compare with first timestep
                gt_first = gt_actions[:, 0, :]
            else:
                gt_first = gt_actions

            # Ensure shapes match for comparison
            if pred_actions.shape != gt_first.shape:
                # Truncate to minimum size if needed
                min_dim = min(pred_actions.shape[-1], gt_first.shape[-1])
                pred_actions = pred_actions[..., :min_dim]
                gt_first = gt_first[..., :min_dim]

            # Compute L1 and L2 loss
            l1_loss = torch.nn.functional.l1_loss(pred_actions, gt_first)
            l2_loss = torch.nn.functional.mse_loss(pred_actions, gt_first)

            l1_acc += l1_loss.item()
            l2_acc += l2_loss.item()
            count += 1

    policy.train()

    if count == 0:
        return {}

    return {
        "loss": l1_acc / count,  # Use L1 as primary loss for early stopping compatibility
        "l1_loss": l1_acc / count,
        "l2_loss": l2_acc / count,
    }


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
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
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    cfg.validate()

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create a separate validation dataset without augmentations if we have validation episodes
    val_dataset = None
    if cfg.validation_fraction > 0:
        # Temporarily disable image transforms for validation dataset
        original_transforms_enable = cfg.dataset.image_transforms.enable
        cfg.dataset.image_transforms.enable = False
        val_dataset = make_dataset(cfg)
        # Restore original setting
        cfg.dataset.image_transforms.enable = original_transforms_enable

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if is_main_process:
            logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    # For SARM, always provide dataset_meta for progress normalization
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
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

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    # Split episodes into train/val if validation_fraction > 0
    val_episode_indices = []
    train_episode_indices = None  # None means use all episodes (default behavior)

    if cfg.validation_fraction > 0:
        num_episodes = dataset.num_episodes
        num_val_episodes = int(num_episodes * cfg.validation_fraction)
        if num_val_episodes == 0:
            logging.warning(
                "Validation fraction is too small to yield any episodes. Using 0 validation episodes."
            )
        else:
            all_indices = list(range(num_episodes))
            if cfg.early_stopping.shuffle_episodes:
                import random
                random.Random(cfg.seed).shuffle(all_indices)

            val_episode_indices = all_indices[:num_val_episodes]
            train_episode_indices = all_indices[num_val_episodes:]

            if is_main_process:
                logging.info(
                    f"Training on {len(train_episode_indices)} episodes, "
                    f"validating on {len(val_episode_indices)} episodes "
                    f"(shuffled={cfg.early_stopping.shuffle_episodes})"
                )

    # Determine if we need to use EpisodeAwareSampler
    # Use sampler when: validation split is active OR policy requires drop_n_last_frames
    use_sampler = train_episode_indices is not None or hasattr(cfg.policy, "drop_n_last_frames")

    if use_sampler:
        # When using validation split, use train_episode_indices
        # Otherwise, use dataset.episodes (respects dataset config) or None (all episodes)
        episode_indices = train_episode_indices if train_episode_indices is not None else dataset.episodes
        train_sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=episode_indices,
            drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
            shuffle=True,
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=train_sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Val dataloader (uses separate dataset without augmentations)
    val_dataloader = None
    if val_episode_indices and val_dataset is not None:
        val_sampler = EpisodeAwareSampler(
            val_dataset.meta.episodes["dataset_from_index"],
            val_dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=val_episode_indices,
            drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
            shuffle=False,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=cfg.num_workers,
            batch_size=1,  # Must be 1 for select_action inference compatibility
            sampler=val_sampler,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
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

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    # Initialize early stopping tracker
    early_stopping_tracker = None
    if cfg.early_stopping.enable:
        early_stopping_tracker = EarlyStoppingTracker(
            patience_steps=cfg.early_stopping.patience_steps,
            min_delta=cfg.early_stopping.min_delta,
            higher_is_better=cfg.early_stopping.higher_is_better,
        )
        if is_main_process:
            logging.info(
                f"Early stopping enabled: patience={cfg.early_stopping.patience_steps}, "
                f"monitor={cfg.early_stopping.monitor}, higher_is_better={cfg.early_stopping.higher_is_better}"
            )

    if is_main_process:
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    early_stop_triggered = False
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log RA-BC statistics if enabled
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if is_eval_step:
            eval_metric_value = None

            if val_dataloader:
                if is_main_process:
                    logging.info(f"Validating on dataset at step {step}")
                val_metrics = validate_dataset_loss(policy, val_dataloader, preprocessor, accelerator)
                if is_main_process:
                    logging.info(f"Validation metrics: {val_metrics}")
                    if wandb_logger:
                        wandb_logger.log_dict({f"val/{k}": v for k, v in val_metrics.items()}, step)

                # Track validation loss for early stopping
                if cfg.early_stopping.monitor == "val_loss" and "loss" in val_metrics:
                    eval_metric_value = val_metrics["loss"]

            if cfg.env:
                if is_main_process:
                    step_id = get_step_identifier(step, cfg.steps)
                    logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # Track success rate for early stopping
                if cfg.early_stopping.monitor == "eval_success":
                    eval_metric_value = aggregated.get("pc_success", 0.0)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            # Early stopping check
            if early_stopping_tracker is not None and eval_metric_value is not None:
                should_stop = early_stopping_tracker.update(eval_metric_value, step)
                status = early_stopping_tracker.get_status()
                if is_main_process:
                    logging.info(
                        f"Early stopping: value={eval_metric_value:.4f}, "
                        f"best={status['best_value']:.4f} @ step {status['best_step']}, "
                        f"no improvement for {status['steps_without_improvement']} steps"
                    )
                if should_stop:
                    if is_main_process:
                        logging.info(
                            f"Early stopping triggered at step {step}. "
                            f"Best value: {status['best_value']:.4f} at step {status['best_step']}"
                        )
                    early_stop_triggered = True

            accelerator.wait_for_everyone()

        # Cleanup old checkpoints if configured
        if cfg.save_checkpoint and is_saving_step and cfg.keep_last_n_checkpoints > 0:
            if is_main_process:
                cleanup_old_checkpoints(cfg.output_dir, cfg.keep_last_n_checkpoints, step)

        # Break out of training loop if early stopping triggered
        if early_stop_triggered:
            break

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        if early_stop_triggered:
            logging.info(f"Training stopped early at step {step} due to early stopping")
        else:
            logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
