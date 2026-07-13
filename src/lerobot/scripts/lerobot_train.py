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

Launch with torchrun for distributed runs; every parallelism/acceleration knob lives on the
config (`--parallelism.*`, `--accelerator.*`) so a run is reproducible from its
train_config.json alone:

```bash
torchrun --nproc-per-node=8 $(which lerobot-train) \
    --dataset.repo_id=... --policy.type=act \
    --parallelism.dp_shard=8 --accelerator.mixed_precision=bf16
```
"""

import dataclasses
import logging
import sys
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
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_batch_size,
    load_training_dp_world_size,
    publish_trained_model,
    push_checkpoint_to_hub,
    resume_after_prepare,
    resume_before_prepare,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import JobConfig, parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, compute_sampler_state
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.distributed import (
    ParallelDims,
    finalize_sharded_policy,
    is_main_process,
    make_accelerator,
    set_fsdp_wrap_modules,
)
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.jobs import submit_to_hf
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.policies.factory import ProcessorConfigKwargs
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
    learning rate scheduler. Accelerator handles mixed-precision training automatically, and — under
    gradient accumulation — suppresses gradient sync on non-final micro-batches and rescales the loss.

    Args:
        train_metrics (MetricsTracker): A MetricsTracker instance to record training statistics.
        policy (PreTrainedPolicy): The policy model to be trained (as returned by `accelerator.prepare`).
        batch (Any): A batch of training data.
        optimizer (Optimizer): The optimizer used to update the policy's parameters.
        grad_clip_norm (float): The maximum norm for gradient clipping (no clipping when <= 0).
        accelerator (Accelerator): The Accelerator instance for distributed training and mixed precision.
        lr_scheduler (LRScheduler | None, optional): An optional learning rate scheduler, stepped once
            per micro-batch. Defaults to None.
        lock (Lock | None, optional): An optional lock for thread-safe optimizer updates.
            Defaults to None.
        sample_weighter (SampleWeighter | None, optional): Optional SampleWeighter instance for
            per-sample loss weighting. Defaults to None.

    Returns:
        tuple[MetricsTracker, dict | None]: The updated MetricsTracker with new statistics for this
        step, and the dictionary of outputs from the policy's forward pass, for logging purposes.
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

    # Under gradient accumulation this context suppresses gradient sync (FSDP2:
    # set_requires_gradient_sync) on non-final micro-batches and divides the loss;
    # with gradient_accumulation_steps == 1 it is a transparent no-op.
    with accelerator.accumulate(policy):
        # Let accelerator handle mixed precision
        with accelerator.autocast():
            # `policy(...)`, never `policy.forward(...)`: FSDP2 all-gathers parameters through
            # nn.Module forward hooks, which only run via __call__.
            if sample_weights is not None:
                # Use per-sample loss for weighted training
                # Note: Policies supporting sample weighting must implement forward(batch, reduction="none")
                per_sample_loss, output_dict = policy(batch, reduction="none")

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
                loss, output_dict = policy(batch)

            # TODO(rcadene): policy.unnormalize_outputs(out_dict)

        # Use accelerator's backward method
        accelerator.backward(loss)

        # Gradients are complete only on sync micro-batches; clipping partial gradients would
        # be meaningless. Always pass the full parameter list: accelerate's FSDP2 path requires
        # an exact match with the prepared model's parameters for a globally correct norm.
        grad_norm = None
        if accelerator.sync_gradients and grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)

        # Optimizer step (a no-op on non-final micro-batches under gradient accumulation)
        with lock if lock is not None else nullcontext():
            optimizer.step()
        optimizer.zero_grad()

        # Step through pytorch scheduler at every batch instead of epoch
        if lr_scheduler is not None:
            lr_scheduler.step()

    # Update internal buffers if policy has update method. These track optimizer updates
    # (EMA, target networks), not micro-batches: gate on the sync step under accumulation.
    if accelerator.sync_gradients and has_method(
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"
    ):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    if grad_norm is not None:
        train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    if torch.cuda.is_available():
        train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return train_metrics, output_dict


def make_dataloaders(
    cfg: TrainPipelineConfig,
    dataset,
    eval_dataset,
    step: int,
    parallel_dims: ParallelDims,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]:
    """Build the train (and optional eval) dataloader, including the sampler resume offset.

    The sampler offset is *derived* from `step` (`resume_before_prepare` loads step + RNG only):
    each loop step consumes `batch_size` samples on each of the `dp_world_size` distinct
    data-parallel workers — no grad-accumulation factor, since `step` counts micro-batches.

    Args:
        cfg (TrainPipelineConfig): The training config (batch size, workers, streaming, resume, seed).
        dataset (LeRobotDataset | MultiLeRobotDataset): The training dataset.
        eval_dataset (LeRobotDataset | None): Optional held-out split; when provided, an eval
            dataloader is built (subsampled per task when `cfg.max_eval_samples > 0`).
        step (int): The loop step to resume the sampler from (0 for a fresh run).
        parallel_dims (ParallelDims): The resolved parallelism topology; provides the device type
            and the fallback dp world size for the resume offset.

    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]: The train
        dataloader and the eval dataloader (None when no eval split exists).
    """
    active_cfg = cfg.trainable_config
    if not cfg.dataset.streaming:
        # All non-streaming (map-style) datasets use EpisodeAwareSampler.
        # The order is a pure function of (seed, epoch), so every rank independently produces the
        # same permutation. accelerate then shards it disjointly across data-parallel ranks via
        # BatchSamplerShard without needing a `generator` attribute to synchronize an RNG, and
        # resume is sample-exact.
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
            shuffle=True,
            seed=cfg.seed if cfg.seed is not None else 0,
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )
        if cfg.resume and step > 0:
            # The resume offset depends on the (dp_world_size, batch_size) that produced `step`,
            # so use the values recorded in the checkpoint (falling back to the current ones for
            # older checkpoints that did not store them).
            saved_dp_world = load_training_dp_world_size(cfg.checkpoint_path)
            saved_batch_size = load_training_batch_size(cfg.checkpoint_path)
            ckpt_dp_world = saved_dp_world or parallel_dims.dp_world_size
            ckpt_batch_size = saved_batch_size or cfg.batch_size
            if is_main_process() and saved_dp_world not in (None, parallel_dims.dp_world_size):
                logging.warning(
                    f"Resuming with dp_world_size={parallel_dims.dp_world_size} but the "
                    f"checkpoint was written with dp_world_size={saved_dp_world}. The data order "
                    "resumes at the right epoch/offset, but per-rank sample-exactness requires "
                    "the same data-parallel world size."
                )
            if is_main_process() and saved_batch_size not in (None, cfg.batch_size):
                logging.warning(
                    f"Resuming with batch_size={cfg.batch_size} but the checkpoint was written "
                    f"with batch_size={saved_batch_size}. The data order resumes at the right "
                    "epoch/offset, but per-rank sample-exactness requires the same batch size."
                )
            sampler_state = compute_sampler_state(step, len(sampler), ckpt_batch_size, ckpt_dp_world)
            sampler.load_state_dict(sampler_state)
            if is_main_process():
                logging.info(
                    f"Resuming data order at epoch {sampler_state['epoch']}, "
                    f"sample {sampler_state['start_index']}"
                )
    else:
        shuffle = True
        sampler = None

    device_type = parallel_dims.device_type
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
        pin_memory=device_type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )

    # Build eval dataloader if a held-out split exists
    eval_dataloader = None
    if eval_dataset is not None:
        eval_ds = eval_dataset
        if cfg.max_eval_samples > 0 and hasattr(eval_dataset, "hf_dataset"):
            task_arr = eval_dataset.hf_dataset.data.column("task_index").to_numpy()
            unique_tasks = sorted(set(task_arr.tolist()))
            per_task = max(1, cfg.max_eval_samples // len(unique_tasks))
            selected: list[int] = []
            for t in unique_tasks:
                frames = (task_arr == t).nonzero()[0][:per_task]
                selected.extend(frames.tolist())
            eval_ds = torch.utils.data.Subset(eval_dataset, selected)

        eval_collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device_type == "cuda",
            drop_last=False,
            collate_fn=eval_collate_fn,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        )
    return dataloader, eval_dataloader


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and the distributed engine.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint (two-phase, around `accelerator.prepare`).
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Publishing the trained model to the Hugging Face Hub if configured.

    Args:
        cfg (TrainPipelineConfig): A `TrainPipelineConfig` object containing all training
            configurations, parsed from the CLI by `parser.wrap()`. On `--resume`, it is the config
            recorded in the checkpoint's `train_config.json`; when `cfg.job.is_remote`, the run is
            dispatched to HF Jobs instead of executing locally.
    """
    if cfg.job.is_remote:
        return submit_to_hf(cfg)

    from lerobot.utils.import_utils import require_package

    require_package("accelerate", extra="training")

    cfg.validate()  # all fail-fasts fire here, before any distributed init

    # --- engine & topology --------------------------------------------------------------------
    # The factory is the ONLY accelerate configuration site: it guards against env-var
    # interference, resolves the declared parallelism degrees against the launched world, and
    # builds the Accelerator from the config mirrors.
    accelerator = make_accelerator(cfg)
    parallel_dims = ParallelDims.from_config(
        cfg.parallelism, accelerator.num_processes, accelerator.device.type
    )
    init_logging(accelerator=accelerator)

    if is_main_process():
        logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project and is_main_process():
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process():
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # --- data (the main process downloads once; peers read the populated cache) ----------------
    if is_main_process():
        logging.info("Creating dataset")
        dataset, eval_dataset = make_train_eval_datasets(cfg)
    accelerator.wait_for_everyone()
    if not is_main_process():
        dataset, eval_dataset = make_train_eval_datasets(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    # (Sharded runs fail fast on env_eval_freq > 0 at validate() time.)
    eval_env = None
    if cfg.env_eval_freq > 0 and cfg.env is not None and is_main_process():
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    # --- policy (weight source decided by the resume rule) -------------------------------------
    # On resume, cfg was parsed FROM the checkpoint's train_config.json, so cfg.checkpoint_format
    # IS the recorded value: DCP-bearing formats skip the safetensors load here and stream the
    # sharded weights in after prepare (resume_after_prepare).
    defer_weight_load = cfg.resume and cfg.checkpoint_format.wants_dcp
    if cfg.is_reward_model_training:
        if is_main_process():
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
        if is_main_process():
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
            defer_weight_load=defer_weight_load,
        )

    peft_model = None
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
        peft_model = policy

    accelerator.wait_for_everyone()

    # --- processors (overrides built once, as one typed mapping) -------------------------------
    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path

    processor_kwargs = ProcessorConfigKwargs()
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

    # Created BEFORE prepare on the unsharded parameters — accelerate's FSDP2 path requires the
    # model and optimizer in one prepare() call and rebinds the param groups itself.
    if is_main_process():
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # --- resume phase 1 + dataloaders ----------------------------------------------------------
    step = 0  # number of loop steps (= micro-batches consumed per data-parallel worker)
    if cfg.resume:
        step = resume_before_prepare(cfg)  # step + RNG only; sharded state loads after prepare

    dataloader, eval_dataloader = make_dataloaders(cfg, dataset, eval_dataset, step, parallel_dims)

    # --- prepare & resume phase 2 ---------------------------------------------------------------
    # The FSDP wrap-unit class names resolve right before prepare: user override, else the
    # policy's _fsdp_wrap_modules declaration — root-only wrapping is never silently accepted.
    set_fsdp_wrap_modules(accelerator, accelerator.unwrap_model(policy) if peft_model else policy)
    accelerator.wait_for_everyone()
    if eval_dataloader is not None:
        policy, optimizer, dataloader, lr_scheduler, eval_dataloader = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler, eval_dataloader
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )
    finalize_sharded_policy(policy, parallel_dims)
    if cfg.resume:
        resume_after_prepare(cfg, accelerator, policy, optimizer, lr_scheduler)

    # --- auxiliaries (after the core assembly, per the construction-order contract) -------------
    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process():
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    # --- banner (main process only; numel() reads metadata — on DTensors it is the GLOBAL shape,
    # so the totals are correct even after sharding) ---------------------------------------------
    # One loop step consumes one micro-batch on every dp worker; the optimizer sees
    # `samples_per_step x gradient_accumulation_steps` samples per update.
    samples_per_step = cfg.batch_size * parallel_dims.dp_world_size
    effective_batch_size = samples_per_step * cfg.accelerator.gradient_accumulation.steps
    if is_main_process():
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
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
        logging.info(
            f"Effective batch size: {cfg.batch_size} x {parallel_dims.dp_world_size} dp workers "
            f"x {cfg.accelerator.gradient_accumulation.steps} grad accum = {effective_batch_size} "
            f"(topology: dp_replicate={parallel_dims.dp_replicate}, dp_shard={parallel_dims.dp_shard})"
        )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    dl_iter = cycle(dataloader)
    policy.train()

    train_metrics = {
        # Per-rank loss reflects only one shard of the global batch; mean recovers the loss the
        # data-parallel group is actually optimizing. grad_norm and lr are already identical on
        # every rank (post gradient sync / deterministic scheduler) so reducing them would be a
        # no-op collective.
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

    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        dp_world_size=parallel_dims.dp_world_size,
    )

    if is_main_process():
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
        if is_main_process():
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_env_eval_step = cfg.env_eval_freq > 0 and step % cfg.env_eval_freq == 0
        is_eval_step = cfg.eval_steps > 0 and eval_dataloader is not None and step % cfg.eval_steps == 0

        if is_log_step:
            # Collective reduce must run on every rank, before the main-process gate below.
            train_tracker.reduce_across_ranks()
            if is_main_process():
                # Cluster-wide throughput, derived from the already-reduced (max) step time so it
                # reflects the slowest rank — which is what actually gates the next iteration.
                step_time = train_tracker.update_s.avg + train_tracker.dataloading_s.avg
                if step_time > 0:
                    train_tracker.samples_per_s = samples_per_step / step_time
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

        if is_eval_step:
            policy.eval()
            eval_loss_sum = 0.0
            n_eval_batches = 0
            with torch.no_grad(), accelerator.autocast():
                for eval_batch in eval_dataloader:
                    for cam_key in dataset.meta.camera_keys:
                        if cam_key in eval_batch and eval_batch[cam_key].dtype == torch.uint8:
                            eval_batch[cam_key] = eval_batch[cam_key].to(dtype=torch.float32) / 255.0
                    eval_batch = preprocessor(eval_batch)
                    loss, _ = policy(eval_batch)  # __call__, so FSDP2 forward hooks run
                    eval_loss_sum += loss.item()
                    n_eval_batches += 1
            eval_loss = eval_loss_sum / max(n_eval_batches, 1)
            eval_loss = torch.tensor(eval_loss, device=device)
            eval_loss = accelerator.reduce(eval_loss, reduction="mean").item()
            policy.train()

            if is_main_process():
                logging.info(f"step {step}: eval_loss={eval_loss:.4f}")
                if wandb_logger:
                    wandb_logger.log_dict({"eval_loss": eval_loss}, step=step, mode="eval")

        if cfg.save_checkpoint and is_saving_step:
            # Collective: every rank participates (gathers / DCP shard writes); rank-0-only file
            # writes are gated inside save_checkpoint — no rank branches at the call site.
            if is_main_process():
                logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                cfg=cfg,
                policy=policy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                accelerator=accelerator,
            )
            if is_main_process():
                update_last_checkpoint(checkpoint_dir)
                if cfg.save_checkpoint_to_hub:
                    push_checkpoint_to_hub(
                        checkpoint_dir,
                        cfg.policy.repo_id,
                        private=cfg.policy.private,
                    )
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)
            accelerator.wait_for_everyone()

        if cfg.env and is_env_eval_step:
            if is_main_process():
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
                    dp_world_size=parallel_dims.dp_world_size,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process():
        progbar.close()
        logging.info("End of training")

    if eval_env:
        close_envs(eval_env)

    # --- publish (collective-safe: all ranks; the model commit gathers sharded weights) ---------
    if getattr(active_cfg, "push_to_hub", False):
        unwrapped = accelerator.unwrap_model(policy)
        model_to_publish = unwrapped.get_base_model() if peft_model is not None else unwrapped
        publish_trained_model(
            cfg,
            model_to_publish,
            preprocessor,
            postprocessor,
            dataset.meta,
            peft_model=unwrapped if peft_model is not None else None,
        )

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def _remote_target_in_argv() -> bool:
    """Detect a remote HF Jobs run request on the raw CLI, before draccus parsing.

    Returns:
        bool: True when the CLI requests a remote HF Jobs run (`--job.target=<non-local>`).
    """
    target = None
    args = sys.argv[1:]
    for i, tok in enumerate(args):
        if tok == "--job.target" and i + 1 < len(args):
            target = args[i + 1]
        elif tok.startswith("--job.target="):
            target = tok.split("=", 1)[1]
    return JobConfig.is_remote_target(target)


def main():
    register_third_party_plugins()
    if _remote_target_in_argv():
        # The policy device is resolved on the remote pod, not here, so silence the
        # client-side "Device '...' is not available" warning PreTrainedConfig emits
        # while parsing the config (it fires before train() can dispatch remotely).
        logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
    train()


if __name__ == "__main__":
    main()
