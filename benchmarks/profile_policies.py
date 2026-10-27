#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Profile fresh-action inference and training updates on one fixed dataset batch.

uv run python benchmarks/profile_policies.py --policy.type=act \
    --dataset.repo_id=lerobot/pusht --dataset.episodes='[0]' \
    --batch_size=1 --policy.device=cuda --output_dir=profiling-results/act
"""

import hashlib
import json
import math
import os
import platform
import runpy
import statistics
import subprocess
import time
import traceback
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import default_collate

from lerobot.configs import parser
from lerobot.configs.parallelism import world_size_from_env
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import make_dataset
from lerobot.datasets.compute_stats import RunningQuantileStats
from lerobot.distributed.factory import make_accelerator
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_batch_keys, rename_stats
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed


@dataclass
class ProfileConfig(TrainPipelineConfig):
    profile_mode: Literal["inference", "training", "both"] = "both"
    warmup: int = 3
    iterations: int = 10
    trace: bool = False
    compute_missing_quantiles: bool = False


def measure(
    step: Callable[[dict[str, Any]], Any],
    batch: dict[str, Any],
    device: torch.device,
    warmup: int,
    iterations: int,
    trace_path: Path | None = None,
    *,
    label: str = "profile",
) -> dict[str, Any]:
    """Keep batch copying, warmup and optional tracing outside latency samples."""
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive")

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    for i in range(warmup):
        print(f"{label}: warmup {i + 1}/{warmup}", flush=True)
        step(deepcopy(batch))
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    allocated, reserved = [], []
    for i in range(iterations):
        print(f"{label}: sample {i + 1}/{iterations}", flush=True)
        sample = deepcopy(batch)
        synchronize()
        start = time.perf_counter()
        step(sample)
        synchronize()
        samples.append(1000 * (time.perf_counter() - start))
        if device.type == "cuda":
            allocated.append(torch.cuda.max_memory_allocated(device))
            reserved.append(torch.cuda.max_memory_reserved(device))
    result = {
        "status": "success",
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
        "peak_allocated_bytes": max(allocated) if allocated else None,
        "peak_reserved_bytes": max(reserved) if reserved else None,
    }
    if trace_path is not None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        sample = deepcopy(batch)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            step(sample)
            synchronize()
        prof.export_chrome_trace(str(trace_path))
    return result


def reset_for_inference(policy: Any) -> None:
    """Generate a fresh chunk while retaining LingBot's fixed per-episode prompt."""
    prompt = None
    if policy.config.type == "lingbot_va" and policy._prompt_embeds is not None:
        prompt = (policy._prompt, policy._prompt_embeds, policy._negative_prompt_embeds)
    policy.reset()
    if prompt is not None:
        policy._prompt, policy._prompt_embeds, policy._negative_prompt_embeds = prompt


def write_result(path: Path, row: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n")
    temporary.replace(path)


def current_observation(batch: dict[str, Any], policy_cfg: Any) -> dict[str, Any]:
    """Select timestamp zero from raw training windows before inference preprocessing."""
    observation = {k: v for k, v in batch.items() if k != "action" and not k.startswith("action_")}
    for key, value in observation.items():
        if not key.startswith("observation.") or key.endswith("_is_pad"):
            continue
        indices = policy_cfg.observation_delta_indices
        if key.startswith("observation.image"):
            indices = getattr(policy_cfg, "image_observation_delta_indices", None) or indices
        elif key == "observation.state":
            indices = getattr(policy_cfg, "state_observation_delta_indices", None) or indices
        feature = (getattr(policy_cfg, "input_features", None) or {}).get(key)
        feature_rank = len(feature.shape) if feature else (3 if key.startswith("observation.image") else 1)
        # The video reader already squeezes a singleton time window; tabular columns retain it.
        if indices is not None and value.ndim == feature_rank + 2:
            observation[key] = value[:, indices.index(0)]
    return observation


def profiling_stats(cfg: ProfileConfig, policy: Any, dataset: Any) -> dict:
    stats = deepcopy(rename_stats(dataset.meta.stats, cfg.rename_map))
    if cfg.compute_missing_quantiles:
        features = {**policy.config.input_features, **policy.config.output_features}
        for source_key in dataset.meta.stats:
            key = (cfg.rename_map or {}).get(source_key, source_key)
            feature = features.get(key)
            if feature is None or policy.config.normalization_mapping.get(feature.type) != "QUANTILES":
                continue
            if all(q in stats[key] for q in ("q01", "q99")):
                continue
            accumulator = RunningQuantileStats(quantile_list=[0.01, 0.99])
            table = dataset.hf_dataset.select_columns([source_key]).with_format("numpy")
            for batch in table.iter(batch_size=4096):
                accumulator.update(np.asarray(batch[source_key]))
            computed = accumulator.get_statistics()
            for q in ("q01", "q99"):
                stats[key].setdefault(q, computed[q])
    return stats


def prepare_policy_config(cfg: ProfileConfig, dataset: Any) -> None:
    if cfg.policy.type == "xvla" and not cfg.policy.pretrained_path and not cfg.policy.florence_config:
        architecture = hf_hub_download(
            "lerobot/xvla-base", "config.json", revision="cdb7964e4fe842935d671bfab5a5ebe00a96648c"
        )
        cfg.policy.florence_config = json.loads(Path(architecture).read_text())["florence_config"]
    if (
        cfg.policy.type == "pi0_fast"
        and cfg.policy.pretrained_path
        and not Path(cfg.policy.pretrained_path).is_dir()
    ):
        # Fail before allocation if the checkpoint is unavailable; the legacy loader can fall back silently.
        weights = hf_hub_download(
            str(cfg.policy.pretrained_path), "model.safetensors", revision=cfg.policy.pretrained_revision
        )
        # The legacy loader does not forward revision consistently. Pin the local snapshot exactly.
        cfg.policy.pretrained_path = Path(weights).parent
    if (
        cfg.policy.type == "flux3"
        and not cfg.policy.pretrained_path
        and cfg.policy.normalization_stats is None
    ):
        # Reuse the documented export's command-delta statistics, rather than raw action quantiles.
        export = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/flux3/export_base.py"))
        cfg.policy.normalization_stats, _ = export["compute_statistics"](
            cfg.dataset.repo_id,
            dataset.meta,
            {"episodes": cfg.dataset.episodes},
            root=dataset.root,
            action_representation=cfg.policy.action_representation,
            absolute_dims=cfg.policy.delta_absolute_dims,
        )


def make_training_step(
    cfg: ProfileConfig, policy: Any, dataset: Any, preprocess: Any, accelerator: Any
) -> Callable:
    """Use the ordinary training update, or SAC for the Gaussian actor."""
    if cfg.policy.type == "gaussian_actor":
        from lerobot.rl.algorithms.sac.configuration_sac import SACAlgorithmConfig
        from lerobot.rl.algorithms.sac.sac_algorithm import SACAlgorithm

        algorithm = SACAlgorithm(policy, SACAlgorithmConfig.from_policy_config(cfg.policy))
        algorithm.make_optimizers_and_scheduler()
        next_indices = [min(i + 1, len(dataset) - 1) for i in range(cfg.batch_size)]
        next_batch = preprocess(default_collate([dataset[i] for i in next_indices]))
        next_state = {key: next_batch[key] for key in policy.config.input_features}
        done = torch.tensor(
            [
                i + 1 >= len(dataset) or dataset[i]["episode_index"] != dataset[j]["episode_index"]
                for i, j in enumerate(next_indices)
            ],
            device=cfg.policy.device,
            dtype=torch.float32,
        )

        def sac_step(batch: dict[str, Any]) -> Any:
            policy.train()
            return algorithm.update(
                iter(
                    repeat(
                        {
                            "state": {key: batch[key] for key in policy.config.input_features},
                            "next_state": next_state,
                            "action": batch["action"],
                            "reward": batch["next.reward"].reshape(-1),
                            "done": batch.get("next.done", done).float().reshape(-1),
                        }
                    )
                )
            )

        return sac_step

    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    policy, optimizer, scheduler = accelerator.prepare(policy, optimizer, scheduler)
    tracker = MetricsTracker(
        cfg.batch_size,
        len(dataset),
        dataset.num_episodes,
        {name: AverageMeter(name) for name in ("loss", "grad_norm", "lr", "update_s")},
    )

    def train_step(batch: dict[str, Any]) -> Any:
        return update_policy(
            tracker, policy, batch, optimizer, cfg.optimizer.grad_clip_norm, accelerator, scheduler
        )

    return train_step


@parser.wrap()
def main(cfg: ProfileConfig) -> None:
    cfg.validate()
    if cfg.policy is None or cfg.output_dir is None:
        raise ValueError("A policy and output_dir are required")
    if cfg.resume or cfg.peft is not None or cfg.dataset.streaming or world_size_from_env() != 1:
        raise ValueError(
            "Profiling uses a single device and a fixed, non-streaming batch without PEFT/resume"
        )
    cfg.output_dir.mkdir(parents=True)
    device = torch.device(cfg.policy.device)
    row: dict[str, Any] = {
        "policy": cfg.policy.type,
        "status": "failed",
        "profile_mode": cfg.profile_mode,
        "gradient_checkpointing": any(
            getattr(cfg.policy, key, False)
            for key in (
                "gradient_checkpointing",
                "use_gradient_checkpointing",
                "vlm_gradient_checkpointing",
                "ae_gradient_checkpointing",
            )
        ),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": subprocess.run(["git", "diff", "--quiet", "HEAD"], check=False).returncode != 0,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "device_total_memory_bytes": (
            torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None
        ),
        "batch_size": cfg.batch_size,
        "warmup": cfg.warmup,
        "iterations": cfg.iterations,
        "dataset": cfg.dataset.repo_id,
        "dataset_revision": cfg.dataset.revision,
        "dataset_episodes": cfg.dataset.episodes,
        "checkpoint": str(cfg.policy.pretrained_path) if cfg.policy.pretrained_path else None,
        "checkpoint_revision": cfg.policy.pretrained_revision,
        "run_url": os.environ.get("PROFILE_RUN_URL"),
    }
    if cfg.policy.type == "lingbot_va":
        row["inference_context"] = "cached_episode_prompt"
    try:
        accelerator = make_accelerator(cfg)
        if accelerator.device.type != device.type:
            raise ValueError(f"Requested {device}, accelerator selected {accelerator.device}")
        set_seed(cfg.seed or 0)
        dataset = make_dataset(cfg)
        prepare_policy_config(cfg, dataset)
        policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
        cfg.save_pretrained(cfg.output_dir)
        saved = json.loads((cfg.output_dir / "train_config.json").read_text())
        row["configuration"] = {key: saved.get(key) for key in ("policy", "optimizer", "accelerator")}
        if row["checkpoint"]:
            row["configuration"]["policy"]["pretrained_path"] = row["checkpoint"]
        row["config_fingerprint"] = hashlib.sha256(
            json.dumps(row["configuration"], sort_keys=True).encode()
        ).hexdigest()
        row["parameter_dtype"] = str(next(policy.parameters()).dtype)
        row["mixed_precision"] = cfg.accelerator.mixed_precision
        # Build processors from the effective policy config and this dataset's statistics.
        preprocessor, _ = make_pre_post_processors(
            policy.config,
            dataset_stats=profiling_stats(cfg, policy, dataset),
            dataset_meta=dataset.meta,
        )
        preprocessor.save_pretrained(cfg.output_dir / "preprocessor")

        def preprocess(batch: dict[str, Any]) -> Any:
            for key in dataset.meta.camera_keys:
                if batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].float() / 255
            return preprocessor(rename_batch_keys(batch, cfg.rename_map))

        raw_batch = default_collate([dataset[i] for i in range(cfg.batch_size)])
        for mode in ("inference", "training"):
            if cfg.profile_mode not in (mode, "both"):
                continue
            try:
                if mode == "inference":
                    batch = preprocess(current_observation(deepcopy(raw_batch), policy.config))

                    def step(batch: dict[str, Any]) -> Any:
                        # Every sample computes a fresh action, including chunk generation.
                        reset_for_inference(policy)
                        policy.eval()
                        with torch.inference_mode(), accelerator.autocast():
                            return policy.select_action(batch)
                else:
                    policy.reset()
                    batch = preprocess(deepcopy(raw_batch))
                    step = make_training_step(cfg, policy, dataset, preprocess, accelerator)
                row[mode] = measure(
                    step,
                    batch,
                    device,
                    cfg.warmup,
                    cfg.iterations,
                    cfg.output_dir / f"{mode}.trace.json" if cfg.trace else None,
                    label=mode,
                )
            except Exception as exc:
                traceback.print_exc()
                row[mode] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            # Preserve completed measurements if a later operation exceeds the job budget.
            write_result(cfg.output_dir / "result.json", row)
        row["status"] = (
            "success"
            if all(row[m]["status"] == "success" for m in ("inference", "training") if m in row)
            else "failed"
        )
    except Exception as exc:
        traceback.print_exc()
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        write_result(cfg.output_dir / "result.json", row)
    if row["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
