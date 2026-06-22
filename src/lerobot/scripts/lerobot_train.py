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
"""Train a policy.

Requires: pip install 'lerobot[training]'  (includes dataset + accelerate + wandb extras)
"""

import dataclasses
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.train_utils import (
    gather_fsdp_state_dicts,
    get_step_checkpoint_dir,
    get_step_identifier,
    load_fsdp_optimizer_state,
    load_training_batch_size,
    load_training_num_processes,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, compute_sampler_state, make_dataset
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

from .lerobot_eval import eval_policy_all


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    sample_weighter=None,
) -> tuple[MetricsTracker, dict | None]:
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
        sample_weighter: Optional SampleWeighter instance for per-sample loss weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Compute sample weights if a weighter is provided
    sample_weights = None
    weight_stats = None
    if sample_weighter is not None:
        sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        if sample_weights is not None:
            # Use per-sample loss for weighted training
            # Note: Policies supporting sample weighting must implement forward(batch, reduction="none")
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Weighted loss: each sample's contribution is scaled by its weight.
            # We divide by weight sum (not batch size) so that if some weights are zero,
            # the remaining samples contribute proportionally more, preserving gradient scale.
            # Weights are pre-normalized to sum to batch_size for stable training dynamics.
            epsilon = 1e-6
            loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)

            # Log weighting statistics
            if output_dict is None:
                output_dict = {}
            for key, value in weight_stats.items():
                output_dict[f"sample_weight_{key}"] = value
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
    if torch.cuda.is_available():
        train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: "Accelerator | None" = None):
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
    from lerobot.utils.import_utils import require_package

    require_package("accelerate", extra="training")
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, DistributedType

    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when the active config's device is set to CPU (works for both policy and reward model training).
        force_cpu = cfg.trainable_config.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

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
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: the global main process downloads once to the shared
    # dataset root, then a barrier lets every other rank read the already-populated copy.
    # LeRobotDataset skips its snapshot_download when try_load() succeeds, so no rank re-downloads.
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Other ranks read from the shared copy populated by the main process.
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if cfg.is_reward_model_training:
        if is_main_process:
            logging.info("Creating reward model")
        from lerobot.rewards import make_reward_model

        policy = make_reward_model(
            cfg=cfg.reward_model,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained via lerobot-train. "
                "Use it directly for inference via compute_reward() (e.g. offline precompute)."
            )
    else:
        if is_main_process:
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
        )

    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        from peft import PeftModel

        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish model creation before continuing
    accelerator.wait_for_everyone()

    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path

    processor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta

    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        preprocessor_overrides = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_overrides = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
        if getattr(active_cfg, "use_relative_actions", False):
            preprocessor_overrides["relative_actions_processor"] = {
                "enabled": True,
                "exclude_joints": getattr(active_cfg, "relative_exclude_joints", []),
                "action_names": getattr(active_cfg, "action_feature_names", None),
            }
            postprocessor_overrides["absolute_actions_processor"] = {"enabled": True}
        processor_kwargs["preprocessor_overrides"] = preprocessor_overrides
        processor_kwargs["postprocessor_overrides"] = postprocessor_overrides

    if cfg.is_reward_model_training:
        preprocessor, postprocessor = make_reward_pre_post_processors(
            cfg.reward_model,
            **processor_kwargs,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=processor_pretrained_path,
            pretrained_revision=getattr(cfg.policy, "pretrained_revision", None),
            **processor_kwargs,
        )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Create sample weighter if configured (e.g., for RA-BC training)
    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process:
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        # Under FSDP the optimizer state is sharded and must be loaded after `accelerator.prepare()`
        # (see load_fsdp_optimizer_state below), so skip the optimizer here and load it then.
        is_fsdp = accelerator.distributed_type == DistributedType.FSDP
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler, load_optimizer=not is_fsdp
        )

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
    if not cfg.dataset.streaming:
        # All non-streaming (map-style) datasets use EpisodeAwareSampler.
        # The order is a pure function of (seed, epoch), so every rank independently produces the
        # same permutation. accelerate then shards it disjointly across ranks via BatchSamplerShard
        # without needing a `generator` attribute to synchronize an RNG, and resume is sample-exact.
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
            shuffle=True,
            seed=cfg.seed if cfg.seed is not None else 0,
        )
        if cfg.resume and step > 0:
            # The resume offset depends on the (num_processes, batch_size) that produced `step`, so
            # use the values recorded in the checkpoint (falling back to the current ones for older
            # ckpts that did not store them).
            saved_num_processes = load_training_num_processes(cfg.checkpoint_path)
            saved_batch_size = load_training_batch_size(cfg.checkpoint_path)
            ckpt_num_processes = saved_num_processes or accelerator.num_processes
            ckpt_batch_size = saved_batch_size or cfg.batch_size
            if is_main_process and saved_num_processes not in (None, accelerator.num_processes):
                logging.warning(
                    f"Resuming with num_processes={accelerator.num_processes} but the checkpoint was "
                    f"written with num_processes={saved_num_processes}. The data order resumes at the "
                    "right epoch/offset, but per-rank sample-exactness requires the same world size."
                )
            if is_main_process and saved_batch_size not in (None, cfg.batch_size):
                logging.warning(
                    f"Resuming with batch_size={cfg.batch_size} but the checkpoint was written with "
                    f"batch_size={saved_batch_size}. The data order resumes at the right epoch/offset, "
                    "but per-rank sample-exactness requires the same batch size."
                )
            sampler_state = compute_sampler_state(step, len(sampler), ckpt_batch_size, ckpt_num_processes)
            sampler.load_state_dict(sampler_state)
            if is_main_process:
                logging.info(
                    f"Resuming data order at epoch {sampler_state['epoch']}, "
                    f"sample {sampler_state['start_index']}"
                )
    else:
        shuffle = True
        sampler = None

    # Only swap in the language-aware collate when the dataset actually
    # declares language columns; otherwise stay on PyTorch's default
    # collate so non-language training runs are unaffected.
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )

    # FSDP optimizer state is sharded across ranks, so it can only be loaded once the optimizer and
    # model are FSDP-wrapped (i.e. after `prepare`). Collective: every rank must participate.
    if cfg.resume and accelerator.distributed_type == DistributedType.FSDP:
        load_fsdp_optimizer_state(policy, optimizer, cfg.checkpoint_path)

    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        # Per-rank loss reflects only one shard of the global batch; mean recovers the loss DDP
        # is actually optimizing. grad_norm and lr are already identical on every rank (post
        # gradient sync / deterministic scheduler) so reducing them would be a no-op collective.
        "loss": AverageMeter("loss", ":.3f", reduction="mean"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        # Report the slowest rank for bottleneck-style timings so multi-GPU runs surface the
        # true straggler instead of rank 0's view.
        "update_s": AverageMeter("updt_s", ":.3f", reduction="max"),
        "dataloading_s": AverageMeter("data_s", ":.3f", reduction="max"),
        # Derived from the post-reduce max step time; set once per log window on the main rank.
        "samples_per_s": AverageMeter("smp/s", ":.0f"),
    }
    if torch.cuda.is_available():
        # max() because headroom is gated by the worst-case rank.
        train_metrics["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f", reduction="max")

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
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
            sample_weighter=sample_weighter,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            # Collective reduce must run on every rank, before the main-process gate below.
            train_tracker.reduce_across_ranks()
            if is_main_process:
                # Cluster-wide throughput, derived from the already-reduced (max) step time so it
                # reflects the slowest rank — which is what actually gates the next iteration.
                step_time = train_tracker.update_s.avg + train_tracker.dataloading_s.avg
                if step_time > 0:
                    train_tracker.samples_per_s = effective_batch_size / step_time
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    # Log sample weighting statistics if enabled
                    if sample_weighter is not None:
                        weighter_stats = sample_weighter.get_stats()
                        wandb_log_dict.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
                    wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            # Under FSDP, gathering the full model + optimizer state dicts is a cross-rank collective,
            # so all ranks must participate; rank 0 then writes the materialized dicts. For DDP /
            # single-GPU the state dicts are saved the normal way inside save_checkpoint.
            is_fsdp = accelerator.distributed_type == DistributedType.FSDP
            if is_fsdp:
                model_state_dict, optim_state_dict = gather_fsdp_state_dicts(policy, optimizer)
            else:
                model_state_dict, optim_state_dict = None, None
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
                    num_processes=accelerator.num_processes,
                    batch_size=cfg.batch_size,
                    model_state_dict=model_state_dict,
                    optim_state_dict=optim_state_dict,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
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

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    model_state_dict = accelerator.get_state_dict(policy) if is_fsdp else None
    if is_main_process:
        logging.info("End of training")

        if getattr(active_cfg, "push_to_hub", False):
            unwrapped_model = accelerator.unwrap_model(policy)
            # PEFT only applies when training a policy — reward models use the plain path.
            if not cfg.is_reward_model_training and cfg.policy.use_peft:
                unwrapped_model.push_model_to_hub(cfg, peft_model=unwrapped_model)
            else:
                unwrapped_model.push_model_to_hub(cfg, state_dict=model_state_dict)
            preprocessor.push_to_hub(active_cfg.repo_id)
            postprocessor.push_to_hub(active_cfg.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
