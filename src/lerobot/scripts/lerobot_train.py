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
import os
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
from lerobot.datasets import (
    EpisodeAwareSampler,
    WeightedEpisodeAwareSampler,
    compute_sampler_state,
    make_dataset,
)
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


def _print_debug_text_predictions(policy: Any, batch: dict[str, Any], step: int, n_samples: int = 5) -> None:
    """Forward the current batch and print head-argmax vs label per supervised position.

    Opt-in via ``LEROBOT_DEBUG_PREDS_EVERY=<step_interval>``. Only the
    policy types that expose ``debug_text_predictions`` participate
    (currently PI052); others are silently skipped. Pretty-prints up to
    ``n_samples`` samples from the current batch, showing the prompt,
    every supervised position's (label, prediction, ✓/✗), and a
    per-sample token-accuracy summary — the cheapest "is text training
    actually learning anything" signal.
    """
    # Accelerator/DDP wraps the policy in a ``module`` attribute and
    # doesn't proxy custom methods through, so a naive
    # ``hasattr(policy, "debug_text_predictions")`` returns False on the
    # wrapper — and the helper would silently no-op. Walk through any
    # ``.module`` indirection (DDP, FSDP, ``accelerator.prepare`` wrappers)
    # to reach the raw policy that actually defines the method.
    inner = policy
    while hasattr(inner, "module") and not hasattr(inner, "debug_text_predictions"):
        inner = inner.module
    if not hasattr(inner, "debug_text_predictions"):
        logging.warning(
            "LEROBOT_DEBUG_PREDS_EVERY set but policy %s has no "
            "debug_text_predictions method — skipping dump.",
            type(inner).__name__,
        )
        return
    try:
        debug = inner.debug_text_predictions(batch, max_samples=n_samples)
    except Exception as exc:  # noqa: BLE001
        logging.warning("debug_text_predictions failed: %s", exc, exc_info=True)
        return
    if not debug:
        logging.warning(
            "debug_text_predictions returned no supervised samples — current batch has no text labels."
        )
        return
    policy = inner  # used below for select_message-style decoding parity

    # Build a tokenizer for decoding — match training side exactly.
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415

        from lerobot.policies.pi052.text_processor_pi052 import (  # noqa: PLC0415
            register_paligemma_loc_tokens,
        )

        tok_name = getattr(policy.config, "tokenizer_name", None) or "google/paligemma-3b-pt-224"
        tokenizer = register_paligemma_loc_tokens(AutoTokenizer.from_pretrained(tok_name))
    except Exception as exc:  # noqa: BLE001
        logging.warning("debug preds: tokenizer load failed: %s", exc)
        return

    ids = debug["input_ids"]
    labels = debug["labels"]
    preds = debug["predictions"]
    attn = debug["attention_mask"]

    n = ids.shape[0]
    print(
        f"\n========== STEP {step} DEBUG PREDICTIONS ({n} samples) ==========",
        flush=True,
    )
    for s in range(n):
        a = attn[s].tolist()
        real = sum(a)
        sid = ids[s].tolist()
        sl = labels[s].tolist()
        sp = preds[s].tolist()
        prompt = tokenizer.decode(sid[:real], skip_special_tokens=False)
        print(f"\n  --- sample {s + 1}/{n} ---", flush=True)
        print(f"  prompt: {prompt!r}", flush=True)

        # Ground-truth target (the contiguous supervised label span).
        sup_ids = [int(sid[i]) for i in range(real) if sl[i] != -100]
        if sup_ids:
            print(
                f"  target  (ground truth)        : {tokenizer.decode(sup_ids, skip_special_tokens=False)!r}",
                flush=True,
            )

        # Training-side teacher-forced argmax on the same prompt+target.
        n_sup = n_ok = 0
        teacher_chars: list[int] = []
        for i in range(1, real):
            label = sl[i]
            if label == -100:
                continue
            n_sup += 1
            pred = int(sp[i - 1])
            teacher_chars.append(pred)
            if label == pred:
                n_ok += 1
        teacher_text = tokenizer.decode(teacher_chars, skip_special_tokens=False) if teacher_chars else ""
        acc = n_ok / max(n_sup, 1)
        print(
            f"  training argmax (teacher-fed) : {teacher_text!r}   acc={n_ok}/{n_sup}={acc:.1%}",
            flush=True,
        )
    print("=" * 60 + "\n", flush=True)


def _build_vqa_oversample_weights(dataset: Any, target_fraction: float) -> "torch.Tensor | None":
    """Build per-frame sampling weights that oversample VQA-annotated frames.

    Scans the dataset's ``language_events`` column for frames carrying a
    ``vqa``-style annotation and returns a weight tensor (length == total
    dataset frames) such that, under multinomial sampling, VQA frames make up
    roughly ``target_fraction`` of the training stream.

    Returns ``None`` (⇒ fall back to uniform episode-aware sampling) when VQA
    frames cannot be detected or there are none.
    """
    if not 0.0 < target_fraction < 1.0:
        logging.warning(
            "vqa_target_fraction must be in (0, 1); got %s — VQA oversampling disabled.",
            target_fraction,
        )
        return None
    hf = getattr(dataset, "hf_dataset", None)
    if hf is None or "language_events" not in getattr(hf, "column_names", []):
        logging.warning("Dataset has no `language_events` column — VQA oversampling disabled.")
        return None

    events_col = hf["language_events"]
    n_frames = len(events_col)
    is_vqa = torch.zeros(n_frames, dtype=torch.bool)
    for i, rows in enumerate(events_col):
        if rows and any((row or {}).get("style") == "vqa" for row in rows):
            is_vqa[i] = True

    n_vqa = int(is_vqa.sum())
    if n_vqa == 0:
        logging.warning("No `vqa` annotations found in the dataset — VQA oversampling disabled.")
        return None
    n_other = n_frames - n_vqa

    # Solve target = (n_vqa·w) / (n_vqa·w + n_other) for the VQA weight w.
    # Clamp to ≥ 1 so VQA frames are never *down*-weighted below uniform.
    weight = (target_fraction * n_other) / ((1.0 - target_fraction) * max(n_vqa, 1))
    weight = max(weight, 1.0)
    weights = torch.ones(n_frames, dtype=torch.double)
    weights[is_vqa] = weight
    logging.info(
        "VQA oversampling: %d/%d frames carry a `vqa` annotation (%.2f%%); "
        "weighting them x%.2f to target ~%.0f%% of the training stream.",
        n_vqa,
        n_frames,
        100.0 * n_vqa / n_frames,
        weight,
        100.0 * target_fraction,
    )
    return weights


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
        from datetime import timedelta

        from accelerate.utils import InitProcessGroupKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Bump the c10d store-get / barrier timeout so the rank-0-only
        # ``make_dataset`` block below doesn't trigger a barrier crash on
        # large datasets. Default is 10 min (``store->get`` 600 s); a
        # 32 k-episode v3 dataset (e.g. ``robocasa_pretrain_human300_v4``)
        # spends >13 min on rank 0 building the episode/frame index
        # while ranks 1-N idle at ``wait_for_everyone()`` and crash with
        # ``DistBackendError: ... wait timeout after 600000ms``. 2 h is
        # plenty of headroom; fast paths are unaffected.
        ipg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when the active config's device is set to CPU (works for both policy and reward model training).
        force_cpu = cfg.trainable_config.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs, ipg_kwargs],
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
    # pi052: even when loading pretrained weights, build the processors
    # from the current pi052 config so the recipe text-label and FAST
    # action-label steps are generated and not silently swapped for the
    # checkpoint's older processor stack.
    if cfg.policy.type == "pi052" and processor_pretrained_path is not None and not cfg.resume:
        logging.warning(
            "pi052 is loading pretrained weights from %s, but building processors from the current "
            "pi052 config so recipe text labels and FAST action labels are generated.",
            processor_pretrained_path,
        )
        processor_pretrained_path = None
    if (
        getattr(active_cfg, "use_relative_actions", False)
        and processor_pretrained_path is not None
        and not cfg.resume
    ):
        logging.warning(
            "use_relative_actions=true with pretrained processors can skip relative transforms if "
            "the checkpoint processors do not define them. Building processors from current policy config."
        )
        processor_pretrained_path = None

    processor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta

    # For pi052 (and any future policy that auto-fits part of its
    # preprocessing per-dataset), pass the dataset repo id so the
    # processor factory can locate/refresh dataset-specific artifacts
    # (e.g. fitted FAST tokenizers per Pertsch et al. 2025 [64],
    # π0.5 §III.C).
    if cfg.policy.type == "pi052":
        processor_kwargs["dataset_repo_id"] = cfg.dataset.repo_id

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
        from_indices = dataset.meta.episodes["dataset_from_index"]
        to_indices = dataset.meta.episodes["dataset_to_index"]
        seed = cfg.seed if cfg.seed is not None else 0

        # When `vqa_target_fraction` is set, oversample VQA-annotated
        # frames via a weighted sampler; otherwise plain episode-aware.
        vqa_weights = None
        if cfg.vqa_target_fraction is not None:
            vqa_weights = _build_vqa_oversample_weights(dataset, cfg.vqa_target_fraction)
        if vqa_weights is not None:
            sampler = WeightedEpisodeAwareSampler(
                from_indices,
                to_indices,
                vqa_weights,
                episode_indices_to_use=dataset.episodes,
                drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
                seed=seed,
            )
        else:
            sampler = EpisodeAwareSampler(
                from_indices,
                to_indices,
                episode_indices_to_use=dataset.episodes,
                drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
                shuffle=True,
                seed=seed,
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

    # ------------------------------------------------------------------
    # EMA setup
    # ------------------------------------------------------------------
    # Shadow copy of the trainable params for late-training averaging
    # (Chi et al. 2023 Diffusion Policy §V.D; openpi JAX trainer ships
    # this with decay=0.999 for pi05_libero; openpi PyTorch port and
    # LeRobot main both skip it). Off by default; opt in with
    # ``--ema.enable=true``. Implemented via ema-pytorch
    # (https://github.com/lucidrains/ema-pytorch) — the standard PyTorch
    # EMA library, also used by lucidrains' diffusion repos.
    ema = None
    if cfg.ema.enable and is_main_process:
        from ema_pytorch import EMA  # noqa: PLC0415

        ema = EMA(
            accelerator.unwrap_model(policy),
            beta=cfg.ema.decay,
            update_after_step=cfg.ema.warmup_steps,
            update_every=1,  # update on every ema.update() call
            # Don't register the live model as an ema submodule — accelerator
            # already owns its lifecycle, and double-registration would
            # double-count its params in ``ema.state_dict()``.
            include_online_model=False,
        )
        ema.to(accelerator.device)
        logging.info(
            "EMA enabled (ema-pytorch): beta=%g, update_after_step=%d, use_for_eval=%s",
            cfg.ema.decay,
            cfg.ema.warmup_steps,
            cfg.ema.use_for_eval,
        )

        # Resume the EMA shadow if a previous run wrote one.
        if cfg.checkpoint_path is not None:
            ema_path = cfg.checkpoint_path / "training_state" / "ema_state.pt"
            if ema_path.exists():
                logging.info("Resuming EMA shadow from %s", ema_path)
                try:
                    ema.load_state_dict(
                        torch.load(ema_path, map_location=accelerator.device, weights_only=True)
                    )
                except Exception as exc:  # noqa: BLE001
                    logging.warning(
                        "Failed to load EMA shadow (%s) — restarting EMA from current live weights",
                        exc,
                    )

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

        # EMA update: pull one step of the live weights into the shadow.
        # Runs only on the main process (the shadow lives there); other
        # ranks rely on the live model staying in sync via accelerator.
        # ``ema-pytorch`` holds an internal reference to the online model
        # (set at construction), so ``ema.update()`` takes no args.
        if ema is not None:
            ema.update()

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        # Optional periodic head-prediction dump for the LM head:
        # ``LEROBOT_DEBUG_PREDS_EVERY=1000`` prints 5 samples + per-token
        # (label, argmax, ✓/✗) every 1000 steps. Cheap diagnostic to see
        # whether the text head is actually learning what we expect, vs
        # collapsing to a fixed token. Refilling the recipe-sample dump
        # budget at the same cadence also redumps the raw input shapes.
        _debug_preds_every = int(os.environ.get("LEROBOT_DEBUG_PREDS_EVERY", "0"))
        if _debug_preds_every > 0 and step % _debug_preds_every == 0 and is_main_process:
            try:
                from lerobot.policies.pi052 import text_processor_pi052 as _tp  # noqa: PLC0415

                _tp._DUMPED_SO_FAR = 0
                _tp._DUMP_BUDGET = max(_tp._DUMP_BUDGET, 5)
            except Exception as exc:  # noqa: BLE001
                logging.debug("Could not reset PI052 debug dump budget: %s", exc, exc_info=True)
            _print_debug_text_predictions(policy, batch, step, n_samples=5)

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
                    # EMA observability: ``ema.step`` is the count of
                    # ``ema.update()`` calls (= optimizer steps once EMA is
                    # enabled); ``ema.initted`` flips to True once we've
                    # crossed ``update_after_step``.
                    if ema is not None:
                        wandb_log_dict["ema/step"] = int(ema.step.item())
                        wandb_log_dict["ema/initted"] = float(ema.initted.item())
                        wandb_log_dict["ema/beta"] = float(cfg.ema.decay)
                    wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        # Periodic W&B example table (camera images + text fields + action endpoints).
        if (
            wandb_logger is not None
            and cfg.wandb.log_examples_freq > 0
            and step % cfg.wandb.log_examples_freq == 0
            and is_main_process
        ):
            try:
                wandb_logger.log_training_examples(
                    batch=batch,
                    step=step,
                    camera_keys=list(dataset.meta.camera_keys),
                    n_samples=cfg.wandb.log_examples_n,
                )
            except Exception as exc:  # noqa: BLE001
                logging.warning("wandb log_training_examples failed: %s", exc)

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
                # Save the EMA shadow alongside the training state so a
                # resumed run picks up exactly where the live EMA left off.
                # ``ema-pytorch.state_dict()`` returns the full shadow
                # nn.Module's state dict + step/initted buffers; saved as
                # .pt (the rest of training_state mixes formats already).
                if ema is not None:
                    try:
                        ema_path = checkpoint_dir / "training_state" / "ema_state.pt"
                        ema_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(ema.state_dict(), ema_path)
                    except Exception as exc:  # noqa: BLE001
                        logging.warning("Failed to save EMA shadow: %s", exc)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                # Use the EMA shadow model for eval when enabled —
                # standard practice for diffusion-style policies (~1–3%
                # lift on closed-loop success). ``ema.ema_model`` is a
                # full nn.Module clone, so we just pass it through; no
                # swap/restore on the live policy needed.
                eval_target_policy = (
                    ema.ema_model
                    if (ema is not None and cfg.ema.use_for_eval)
                    else accelerator.unwrap_model(policy)
                )
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=eval_target_policy,
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

            # When EMA is on we *eval* the EMA weights but the push above
            # ships the live weights — they're different models. Push the EMA
            # weights too, to a sibling ``<repo_id>-ema`` repo, so both are
            # fully loadable and you can benchmark/deploy whichever is better.
            # Non-fatal: the live model is already up if this fails.
            if ema is not None and not (
                not cfg.is_reward_model_training and cfg.policy.use_peft
            ):
                ema_model = ema.ema_model
                ema_repo_id = f"{active_cfg.repo_id}-ema"
                orig_repo_id = ema_model.config.repo_id
                try:
                    ema_model.config.repo_id = ema_repo_id
                    ema_model.push_model_to_hub(cfg)
                    preprocessor.push_to_hub(ema_repo_id)
                    postprocessor.push_to_hub(ema_repo_id)
                    logging.info("Pushed EMA weights to %s", ema_repo_id)
                except Exception as exc:  # noqa: BLE001
                    logging.warning("Failed to push EMA weights to %s: %s", ema_repo_id, exc)
                finally:
                    ema_model.config.repo_id = orig_repo_id

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
