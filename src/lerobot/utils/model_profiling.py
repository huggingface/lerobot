#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Model profiling — single-file entry point.

Contains three things that used to live in three separate files:

* `TrainingProfiler` — hooks the training loop. Captures per-step
  forward/backward/optimizer timings, the torch profiler output, and a
  deterministic-forward fingerprint for regression detection.
* `POLICY_SPECS` — CI matrix of `policy_name → (steps, train_args)`.
  Inline so there is no separate JSON to keep in sync.
* `main()` — CI orchestrator. For each selected policy, spawns a
  `lerobot-train` subprocess with profiling enabled, collects the
  artifacts, and (optionally) publishes a row to a HF Hub dataset.

Usage (CI):

    python -m lerobot.utils.model_profiling \
        --output_dir=./profiling-results \
        --policies act diffusion \
        --profile_mode=trace \
        --publish
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import statistics
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError
from torch.utils.data import default_collate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy matrix. Same shape as the former JSON file; inlined so the source
# tree has one less file to keep in sync with the training args.
# ---------------------------------------------------------------------------

_LIBERO_RENAME_BASE_RGB = (
    '--rename_map={"observation.images.front": "observation.images.base_0_rgb", '
    '"observation.images.wrist": "observation.images.left_wrist_0_rgb"}'
)
_LIBERO_RENAME_CAMERAS = (
    '--rename_map={"observation.images.front": "observation.images.camera1", '
    '"observation.images.wrist": "observation.images.camera2"}'
)
_PI_SGD = [
    "--use_policy_training_preset=false",
    "--optimizer.type=sgd",
    "--optimizer.lr=1e-5",
    "--optimizer.weight_decay=0",
    "--optimizer.grad_clip_norm=1.0",
    "--scheduler.type=cosine_decay_with_warmup",
    "--scheduler.peak_lr=1e-5",
    "--scheduler.decay_lr=1e-6",
    "--scheduler.num_warmup_steps=0",
    "--scheduler.num_decay_steps=12",
]


POLICY_SPECS: dict[str, dict[str, Any]] = {
    "act": {
        "steps": 12,
        "train_args": [
            "--dataset.repo_id=lerobot/pusht",
            "--dataset.episodes=[0]",
            "--policy.type=act",
            "--policy.device=cuda",
            "--batch_size=4",
            "--cudnn_deterministic=true",
        ],
    },
    "diffusion": {
        "steps": 12,
        "train_args": [
            "--dataset.repo_id=lerobot/pusht",
            "--dataset.episodes=[0]",
            "--policy.type=diffusion",
            "--policy.device=cuda",
            "--batch_size=4",
            "--cudnn_deterministic=true",
        ],
    },
    "groot": {
        "steps": 12,
        "train_args": [
            "--dataset.repo_id=lerobot/libero_plus",
            "--dataset.episodes=[0]",
            "--policy.type=groot",
            "--policy.base_model_path=nvidia/GR00T-N1.5-3B",
            "--policy.tune_diffusion_model=true",
            "--policy.tune_projector=true",
            "--policy.tune_llm=false",
            "--policy.tune_visual=false",
            "--policy.use_bf16=true",
            "--policy.device=cuda",
            "--batch_size=1",
            '--rename_map={"observation.images.image": "observation.images.camera1", '
            '"observation.images.image2": "observation.images.camera2"}',
        ],
    },
    "multi_task_dit": {
        "steps": 12,
        "train_args": [
            "--dataset.repo_id=lerobot/pusht",
            "--dataset.episodes=[0]",
            "--policy.type=multi_task_dit",
            "--policy.device=cuda",
            "--policy.horizon=32",
            "--policy.n_action_steps=30",
            "--batch_size=4",
            "--cudnn_deterministic=true",
        ],
    },
    "pi0": {
        "steps": 12,
        "train_args": [
            "--dataset.repo_id=lerobot/libero_plus",
            "--dataset.episodes=[0]",
            "--policy.path=lerobot/pi0_base",
            "--policy.device=cuda",
            "--policy.dtype=bfloat16",
            "--policy.n_action_steps=30",
            "--policy.use_amp=true",
            "--policy.gradient_checkpointing=true",
            "--batch_size=1",
            *_PI_SGD,
            _LIBERO_RENAME_BASE_RGB,
        ],
    },
    "pi0_fast": {
        "steps": 12,
        "train_args": [
            "--dataset.repo_id=lerobot/libero_plus",
            "--dataset.episodes=[0]",
            "--policy.path=lerobot/pi0fast-base",
            "--policy.device=cuda",
            "--policy.dtype=bfloat16",
            "--policy.n_action_steps=30",
            "--policy.use_amp=true",
            "--policy.gradient_checkpointing=true",
            "--batch_size=1",
            *_PI_SGD,
            _LIBERO_RENAME_BASE_RGB,
        ],
    },
    "pi05": {
        "steps": 12,
        "train_args": [
            "--dataset.repo_id=lerobot/libero_plus",
            "--dataset.episodes=[0]",
            "--policy.path=lerobot/pi05_base",
            "--policy.device=cuda",
            "--policy.dtype=bfloat16",
            "--policy.n_action_steps=30",
            "--policy.use_amp=true",
            "--policy.gradient_checkpointing=true",
            "--batch_size=1",
            *_PI_SGD,
            '--policy.normalization_mapping={"ACTION": "MEAN_STD", '
            '"STATE": "MEAN_STD", "VISUAL": "IDENTITY"}',
            _LIBERO_RENAME_BASE_RGB,
        ],
    },
    "smolvla": {
        "steps": 12,
        "train_args": [
            "--dataset.repo_id=lerobot/libero_plus",
            "--dataset.episodes=[0]",
            "--policy.path=lerobot/smolvla_base",
            "--policy.load_vlm_weights=true",
            "--policy.freeze_vision_encoder=false",
            "--policy.train_expert_only=false",
            "--policy.empty_cameras=1",
            "--policy.device=cuda",
            "--batch_size=1",
            _LIBERO_RENAME_CAMERAS,
        ],
    },
    "wall_x": {
        "steps": 12,
        "train_args": [
            "--dataset.repo_id=lerobot/aloha_sim_insertion_human",
            "--dataset.episodes=[0]",
            "--policy.type=wall_x",
            "--policy.pretrained_name_or_path=x-square-robot/wall-oss-flow",
            "--policy.prediction_mode=diffusion",
            "--policy.attn_implementation=eager",
            "--policy.device=cuda",
            "--batch_size=1",
            *_PI_SGD,
        ],
    },
    "xvla": {
        "steps": 12,
        "train_args": [
            "--dataset.repo_id=lerobot/libero_plus",
            "--dataset.episodes=[0]",
            "--policy.path=lerobot/xvla-widowx",
            "--policy.action_mode=auto",
            "--policy.empty_cameras=1",
            "--policy.device=cuda",
            "--batch_size=1",
            '--rename_map={"observation.images.front": "observation.images.image", '
            '"observation.images.wrist": "observation.images.image2"}',
        ],
    },
}


# ---------------------------------------------------------------------------
# TrainingProfiler — hooks the training loop.
# ---------------------------------------------------------------------------


def _stable_float(value: float | int | None) -> float | None:
    return None if value is None else round(float(value), 8)


def _as_float(value: Any) -> float:
    if isinstance(value, Real):
        return float(value)
    if hasattr(value, "val"):
        return float(value.val)
    raise TypeError(f"Expected a real-valued metric, got {type(value).__name__}")


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _tensor_signature(tensor: torch.Tensor) -> dict[str, Any]:
    """Small, stable summary of a tensor so forward-pass outputs can be
    compared across runs without bloating the regression JSON."""
    cpu = tensor.detach().cpu()
    hash_tensor = cpu.float() if cpu.dtype == torch.bfloat16 else cpu
    sig: dict[str, Any] = {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "numel": cpu.numel(),
        "sha256": hashlib.sha256(hash_tensor.contiguous().numpy().tobytes()).hexdigest(),
    }
    if cpu.numel():
        promoted = cpu.to(torch.float64) if cpu.is_floating_point() else cpu.to(torch.int64)
        sig["sum"] = _stable_float(promoted.sum().item())
        sig["mean"] = _stable_float(promoted.float().mean().item())
    return sig


def _summarize_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_signature(value)
    if isinstance(value, dict):
        return {k: _summarize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _get_profiler_device_time_us(event: Any) -> float | None:
    return _stable_float(
        getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", None))
    )


def _write_profiler_table(profiler: Any, path: Path, *, sort_by: str, row_limit: int = 40) -> None:
    try:
        path.write_text(profiler.key_averages().table(sort_by=sort_by, row_limit=row_limit))
    except Exception:
        logger.debug("Could not write profiler table for sort_by=%s", sort_by, exc_info=True)


def write_deterministic_forward_artifacts(
    *,
    policy: Any,
    dataset: Any,
    batch_size: int,
    preprocessor: Any,
    output_dir: Path,
    device_type: str,
) -> None:
    """Run a seed-controlled single forward pass and dump a stable fingerprint
    (loss/output tensor hashes + op counts) for regression detection. Keeps
    the caller-selected module mode so ACT-with-VAE-style policies that only
    materialize their full forward outputs in `train()` still match. Models
    with stochastic train-mode layers still rely on the seeded RNG for stable
    fingerprints."""
    if len(dataset) == 0:
        raise ValueError("Cannot build a reference batch from an empty dataset.")
    indices = [i % len(dataset) for i in range(batch_size)]
    reference_batch = default_collate([dataset[i] for i in indices])
    # Mirror the uint8 → float32/255 conversion the train loop applies after
    # the dataloader (PR #3406). The dataset ships camera frames as uint8 for
    # faster transport, but policies like SmolVLA/xVLA run bilinear
    # interpolation on images which doesn't support Byte tensors.
    camera_keys = tuple(getattr(getattr(dataset, "meta", None), "camera_keys", ()) or ())
    if not camera_keys:
        camera_keys = tuple(
            key
            for key, value in reference_batch.items()
            if key.startswith("observation.images.") and isinstance(value, torch.Tensor)
        )
    for cam_key in camera_keys:
        if cam_key in reference_batch and reference_batch[cam_key].dtype == torch.uint8:
            reference_batch[cam_key] = reference_batch[cam_key].to(dtype=torch.float32) / 255.0
    reference_batch = preprocessor(reference_batch)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device_type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.random.fork_rng(devices=[] if device_type != "cuda" else None):
        torch.manual_seed(0)
        if device_type == "cuda":
            torch.cuda.manual_seed_all(0)
        with torch.no_grad(), torch.profiler.profile(activities=activities) as prof:
            loss, output_dict = policy.forward(reference_batch)

    operators = sorted(
        (
            {
                "key": e.key,
                "count": e.count,
                "cpu_time_total_us": _stable_float(getattr(e, "cpu_time_total", None)),
                **(
                    {"self_cuda_time_total_us": _get_profiler_device_time_us(e)}
                    if device_type == "cuda"
                    else {}
                ),
            }
            for e in prof.key_averages()
        ),
        key=lambda e: e["key"],
    )
    outputs = {"loss": _summarize_value(loss), "output_dict": _summarize_value(output_dict)}
    payload = {
        "seed": 0,
        "reference_batch_size": batch_size,
        "operator_fingerprint": _hash_payload([(o["key"], o["count"]) for o in operators]),
        "output_fingerprint": _hash_payload(outputs),
        "operators": operators,
        "outputs": outputs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "deterministic_forward.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    sort_by = "self_cuda_time_total" if device_type == "cuda" else "cpu_time_total"
    _write_profiler_table(prof, output_dir / "deterministic_forward_ops.txt", sort_by=sort_by)


class TrainingProfiler:
    """Self-contained profiling hooks for the training loop.

    The training script interacts via ``start()``, ``section()``, ``step()``,
    ``finalize()``, and (optionally) ``record_deterministic_forward()`` — a
    ~7-line surface.
    """

    _SCHEDULE_WAIT = 1
    _SCHEDULE_WARMUP = 2
    _SCHEDULE_ACTIVE = 6

    def __init__(self, mode: str, output_dir: Path, device: torch.device) -> None:
        self._mode = mode
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._device = device
        # Inline timing state — no separate collector class.
        self._total_update_s: list[float] = []
        self._dataloading_s: list[float] = []
        self._section_s: dict[str, list[float]] = {}
        self._memory: list[dict[str, int]] = []
        self._torch = self._build_torch_profiler()
        logger.info("Profiling enabled. Artifacts will be written to %s", output_dir)

    def _build_torch_profiler(self) -> Any:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self._device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        trace_dir = self._output_dir / "torch_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)

        def _on_trace_ready(p: Any) -> None:
            if self._mode == "trace":
                p.export_chrome_trace(str(trace_dir / f"trace_step_{p.step_num}.json"))

        return torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self._SCHEDULE_WAIT,
                warmup=self._SCHEDULE_WARMUP,
                active=self._SCHEDULE_ACTIVE,
                repeat=1,
            ),
            on_trace_ready=_on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        )

    @classmethod
    def from_cfg(cls, cfg: Any, device: torch.device) -> TrainingProfiler:
        output = cfg.profile_output_dir or (Path(cfg.output_dir) / "profiling")
        return cls(mode=cfg.profile_mode, output_dir=Path(output), device=device)

    def record_deterministic_forward(
        self, *, policy: Any, dataset: Any, batch_size: int, preprocessor: Any
    ) -> None:
        logger.info("Recording deterministic forward-pass artifacts")
        write_deterministic_forward_artifacts(
            policy=policy,
            dataset=dataset,
            batch_size=batch_size,
            preprocessor=preprocessor,
            output_dir=self._output_dir,
            device_type=self._device.type,
        )
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    def start(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
        self._torch.__enter__()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Time a region of the training step. Syncs on CUDA so the
        duration reflects GPU work, not just kernel-launch latency."""
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            self._section_s.setdefault(name, []).append(time.perf_counter() - t0)

    def step(self, step_num: int, train_tracker: Any) -> None:
        self._total_update_s.append(_as_float(train_tracker.update_s))
        self._dataloading_s.append(_as_float(train_tracker.dataloading_s))
        if self._device.type == "cuda":
            self._memory.append(
                {
                    "step": step_num,
                    "allocated_bytes": torch.cuda.memory_allocated(self._device),
                    "reserved_bytes": torch.cuda.memory_reserved(self._device),
                }
            )
        self._torch.step()

    def finalize(self) -> None:
        self._torch.__exit__(None, None, None)
        payload: dict[str, Any] = {
            "profile_mode": self._mode,
            "total_update_s": _summary(self._total_update_s),
            "dataloading_s": _summary(self._dataloading_s),
            "memory_timeline": self._memory,
        }
        for name, values in self._section_s.items():
            payload[f"{name}_s"] = _summary(values)
        if self._device.type == "cuda":
            payload["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(self._device)
            payload["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(self._device)
        (self._output_dir / "step_timing_summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True)
        )

        tables_dir = self._output_dir / "torch_tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        _write_profiler_table(self._torch, tables_dir / "cpu_time_total.txt", sort_by="cpu_time_total")
        _write_profiler_table(self._torch, tables_dir / "cpu_memory.txt", sort_by="self_cpu_memory_usage")
        _write_profiler_table(self._torch, tables_dir / "flops.txt", sort_by="flops")
        if self._device.type == "cuda":
            _write_profiler_table(
                self._torch, tables_dir / "cuda_time_total.txt", sort_by="self_cuda_time_total"
            )
            _write_profiler_table(
                self._torch, tables_dir / "cuda_memory.txt", sort_by="self_cuda_memory_usage"
            )


# ---------------------------------------------------------------------------
# CI orchestrator. Spawns `lerobot-train` per policy, collects the
# artifacts, (optionally) uploads to the HF Hub results dataset.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UploadTarget:
    local_path: Path
    path_in_repo: str


@dataclass(frozen=True)
class UploadResult:
    uploaded_paths: dict[str, str]
    pr_url: str | None = None


def _utc_timestamp_slug(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")


def _hub_file_url(repo_id: str, path_in_repo: str, *, revision: str = "main") -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path_in_repo}"


def parse_discussion_num(pr_url: str | None) -> int | None:
    if not pr_url:
        return None
    m = re.search(r"/discussions/(\d+)$", pr_url)
    return int(m.group(1)) if m else None


def upload_targets(
    repo_id: str,
    targets: list[UploadTarget],
    *,
    token: str | None = None,
    commit_message: str | None = None,
    create_pr: bool = False,
) -> UploadResult:
    api = HfApi(token=token)
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=t.path_in_repo, path_or_fileobj=str(t.local_path))
            for t in targets
        ],
        commit_message=commit_message or f"Upload {len(targets)} profiling artifacts",
        revision="main",
        create_pr=create_pr,
    )
    pr_num = parse_discussion_num(commit.pr_url)
    revision = f"refs/pr/{pr_num}" if (create_pr and pr_num) else "main"
    return UploadResult(
        uploaded_paths={
            t.path_in_repo: _hub_file_url(repo_id, t.path_in_repo, revision=revision) for t in targets
        },
        pr_url=commit.pr_url,
    )


def build_train_command(policy: str, run_dir: Path, profile_mode: str) -> list[str]:
    spec = POLICY_SPECS[policy]
    return [
        "uv",
        "run",
        "lerobot-train",
        *spec["train_args"],
        f"--output_dir={run_dir / 'train'}",
        f"--steps={spec['steps']}",
        "--eval_freq=0",
        "--save_checkpoint=false",
        f"--save_freq={spec['steps']}",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
        "--num_workers=0",
        "--log_freq=1",
        f"--profile_mode={profile_mode}",
        f"--profile_output_dir={run_dir / 'profiling'}",
    ]


def build_artifact_index(
    *, repo_id: str, run_dir: Path, policy_name: str, run_id: str
) -> tuple[dict[str, Any], dict[str, Any], list[UploadTarget], str]:
    """Scan the run directory and categorize files into
    (stdout/stderr, torch_tables/*, torch_traces/*, everything else under profiling/).
    Returns (paths, urls, upload targets, row path in repo)."""
    row_path_in_repo = f"rows/{policy_name}/{run_id}.json"
    root = f"artifacts/{policy_name}/{run_id}"
    paths: dict[str, Any] = {
        "row": row_path_in_repo,
        "profiling_files": {},
        "torch_tables": {},
        "trace_files": {},
    }
    urls: dict[str, Any] = {
        "row": _hub_file_url(repo_id, row_path_in_repo),
        "profiling_files": {},
        "torch_tables": {},
        "trace_files": {},
    }
    targets: list[UploadTarget] = []

    for name in ("stdout.txt", "stderr.txt"):
        p = run_dir / name
        if p.exists():
            key = name.removesuffix(".txt")
            repo = f"{root}/{name}"
            paths[key] = repo
            urls[key] = _hub_file_url(repo_id, repo)
            targets.append(UploadTarget(p, repo))

    profiling_dir = run_dir / "profiling"
    if profiling_dir.exists():
        for p in sorted(profiling_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(run_dir))
            repo = f"{root}/{rel}"
            paths["profiling_files"][rel] = repo
            urls["profiling_files"][rel] = _hub_file_url(repo_id, repo)
            targets.append(UploadTarget(p, repo))
            if p.name == "step_timing_summary.json":
                paths["step_timing_summary"] = repo
                urls["step_timing_summary"] = _hub_file_url(repo_id, repo)
            elif "torch_tables" in p.parts:
                paths["torch_tables"][p.name] = repo
                urls["torch_tables"][p.name] = _hub_file_url(repo_id, repo)
            elif "torch_traces" in p.parts:
                paths["trace_files"][p.name] = repo
                urls["trace_files"][p.name] = _hub_file_url(repo_id, repo)

    return paths, urls, targets, row_path_in_repo


def upload_profile_run(
    *,
    repo_id: str,
    row_path: Path,
    row_path_in_repo: str,
    artifact_targets: list[UploadTarget],
    create_pr: bool = False,
) -> UploadResult:
    return upload_targets(
        repo_id=repo_id,
        targets=[*artifact_targets, UploadTarget(row_path, row_path_in_repo)],
        commit_message=f"Add model profiling row {row_path_in_repo}",
        create_pr=create_pr,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", nargs="*", default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hub_org", default="lerobot")
    parser.add_argument("--results_repo", default="model-profiling-history")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--profile_mode", choices=["summary", "trace"], default="trace")
    parser.add_argument("--git_commit", default="")
    parser.add_argument("--git_ref", default="")
    parser.add_argument("--pr_number", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = args.policies or list(POLICY_SPECS)
    unknown = sorted(set(selected) - set(POLICY_SPECS))
    if unknown:
        raise ValueError(f"Unknown profiling policies: {', '.join(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo_id = args.results_repo if "/" in args.results_repo else f"{args.hub_org}/{args.results_repo}"
    git_exe = shutil.which("git")
    if not git_exe:
        raise RuntimeError("git not found in PATH")
    git_commit = args.git_commit or subprocess.check_output([git_exe, "rev-parse", "HEAD"], text=True).strip()
    pr_number = int(args.pr_number) if str(args.pr_number).strip() else None
    exit_code = 0

    for policy in selected:
        run_id = f"{_utc_timestamp_slug()}__{policy}"
        run_dir = args.output_dir / policy / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_train_command(policy, run_dir, args.profile_mode)

        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        wall_s = time.perf_counter() - t0

        (run_dir / "stdout.txt").write_text(result.stdout)
        (run_dir / "stderr.txt").write_text(result.stderr)
        if result.returncode != 0:
            exit_code = 1

        paths, urls, upload_list, row_in_repo = build_artifact_index(
            repo_id=repo_id, run_dir=run_dir, policy_name=policy, run_id=run_id
        )
        row: dict[str, Any] = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "policy": policy,
            "git_commit": git_commit,
            "git_ref": args.git_ref or None,
            "pr_number": pr_number,
            "status": "success" if result.returncode == 0 else "failed",
            "return_code": result.returncode,
            "profile_mode": args.profile_mode,
            "wall_time_s": wall_s,
            "spec": {
                "steps": POLICY_SPECS[policy]["steps"],
                "train_args": POLICY_SPECS[policy]["train_args"],
            },
            "step_timing_summary": _load_json(run_dir / "profiling" / "step_timing_summary.json"),
            "deterministic_forward": _load_json(run_dir / "profiling" / "deterministic_forward.json"),
            "artifact_paths": paths,
            "artifact_urls": urls,
            "stderr_tail": result.stderr.splitlines()[-20:],
        }

        row_path = run_dir / "profiling_row.json"
        row_path.write_text(json.dumps(row, indent=2, sort_keys=True))

        if args.publish:
            try:
                uploaded = upload_profile_run(
                    repo_id=repo_id,
                    row_path=row_path,
                    row_path_in_repo=row_in_repo,
                    artifact_targets=upload_list,
                    create_pr=pr_number is not None,
                )
            except HfHubHTTPError as exc:
                row.update({"publish_status": "failed", "publish_error": str(exc)})
            else:
                row.update(
                    {
                        "publish_status": "success",
                        "uploaded_paths": uploaded.uploaded_paths,
                        "publish_pr_url": uploaded.pr_url,
                        "publish_pr_number": parse_discussion_num(uploaded.pr_url),
                    }
                )
            row_path.write_text(json.dumps(row, indent=2, sort_keys=True))

        print(json.dumps(row, indent=2, sort_keys=True))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
