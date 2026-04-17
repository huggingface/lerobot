#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Generate lightweight SLURM jobs for policy x benchmark benchmarking."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lerobot.utils.history_repo import utc_timestamp_slug

MAX_GPUS = 8
MIN_GPUS = 1
DEFAULT_STEPS = 20_000
DEFAULT_EFFECTIVE_BATCH_SIZE = 256
DEFAULT_MICROBATCH_PER_GPU = 32
DEFAULT_EVAL_BATCH_SIZE = 1
DEFAULT_CPUS_PER_GPU = 8
DEFAULT_MEMORY_PER_GPU_GB = 40


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset_repo_id: str
    docker_image: str
    eval_env_type: str
    eval_task: str
    eval_n_episodes: int
    train_steps: int = DEFAULT_STEPS
    effective_batch_size: int = DEFAULT_EFFECTIVE_BATCH_SIZE
    train_extra_args: dict[str, Any] = field(default_factory=dict)
    eval_extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    policy_type: str
    num_gpus: int
    policy_path: str | None = None
    microbatch_per_gpu: int = DEFAULT_MICROBATCH_PER_GPU
    extra_train_args: dict[str, Any] = field(default_factory=dict)
    extra_eval_args: dict[str, Any] = field(default_factory=dict)
    needs_tokenizer: bool = False
    tokenizer_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedJob:
    benchmark: str
    policy: str
    run_rel: str
    num_gpus: int
    microbatch_per_gpu: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    docker_image: str
    train_args: dict[str, Any]
    eval_args: dict[str, Any]
    tokenizer_args: dict[str, Any] | None
    script_path: str


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "libero_plus": BenchmarkSpec(
        name="libero_plus",
        dataset_repo_id="lerobot/libero_plus",
        docker_image="lerobot-benchmark-libero-plus:latest",
        eval_env_type="libero_plus",
        eval_task="libero_spatial,libero_object,libero_goal,libero_10",
        eval_n_episodes=10,
        train_extra_args={
            "rename_map": {
                "observation.images.image": "observation.images.camera1",
                "observation.images.image2": "observation.images.camera2",
            },
        },
        eval_extra_args={
            "env.camera_name_mapping": {
                "agentview_image": "camera1",
                "robot0_eye_in_hand_image": "camera2",
            },
            "env.max_parallel_tasks": 1,
            "eval.batch_size": DEFAULT_EVAL_BATCH_SIZE,
            "eval.use_async_envs": False,
            "eval.max_episodes_rendered": 0,
            "policy.device": "cuda",
        },
    ),
    "robomme": BenchmarkSpec(
        name="robomme",
        dataset_repo_id="lerobot/robomme",
        docker_image="lerobot-benchmark-robomme:latest",
        eval_env_type="robomme",
        eval_task=(
            "BinFill,PickXtimes,SwingXtimes,StopCube,VideoUnmask,VideoUnmaskSwap,"
            "ButtonUnmask,ButtonUnmaskSwap,PickHighlight,VideoRepick,VideoPlaceButton,"
            "VideoPlaceOrder,MoveCube,InsertPeg,PatternLock,RouteStick"
        ),
        eval_n_episodes=50,
        train_extra_args={
            "rename_map": {
                "observation.images.image": "observation.images.camera1",
                "observation.images.wrist_image": "observation.images.camera2",
            },
        },
        eval_extra_args={
            "env.dataset_split": "test",
            "env.max_parallel_tasks": 1,
            "rename_map": {
                "observation.images.image": "observation.images.camera1",
                "observation.images.wrist_image": "observation.images.camera2",
            },
            "eval.batch_size": DEFAULT_EVAL_BATCH_SIZE,
            "eval.use_async_envs": False,
            "eval.max_episodes_rendered": 0,
            "policy.device": "cuda",
        },
    ),
}


POLICIES: dict[str, PolicySpec] = {
    "pi0": PolicySpec(
        name="pi0",
        policy_type="pi0",
        policy_path="lerobot/pi0_base",
        num_gpus=8,
        extra_train_args={
            "policy.n_action_steps": 30,
            "policy.scheduler_decay_steps": DEFAULT_STEPS,
            "policy.empty_cameras": 0,
        },
    ),
    "pi0_fast": PolicySpec(
        name="pi0_fast",
        policy_type="pi0_fast",
        policy_path="lerobot/pi0fast-base",
        num_gpus=8,
        extra_train_args={
            "policy.n_action_steps": 30,
            "policy.scheduler_decay_steps": DEFAULT_STEPS,
            "policy.empty_cameras": 0,
        },
        needs_tokenizer=True,
        tokenizer_args={
            "action_horizon": 30,
            "encoded_dims": "0:7",
            "normalization_mode": "QUANTILES",
            "vocab_size": 1024,
            "scale": 10.0,
            "push_to_hub": True,
        },
    ),
    "pi05": PolicySpec(
        name="pi05",
        policy_type="pi05",
        policy_path="lerobot/pi05_base",
        num_gpus=8,
        extra_train_args={
            "policy.n_action_steps": 30,
            "policy.scheduler_decay_steps": DEFAULT_STEPS,
            "policy.empty_cameras": 0,
        },
    ),
    "groot": PolicySpec(
        name="groot",
        policy_type="groot",
        num_gpus=8,
        extra_train_args={
            "policy.n_action_steps": 30,
            "policy.base_model_path": "nvidia/GR00T-N1.5-3B",
            "policy.tune_diffusion_model": True,
            "policy.tune_projector": True,
            "policy.tune_llm": False,
            "policy.tune_visual": False,
            "policy.use_bf16": True,
        },
    ),
    "act": PolicySpec(
        name="act",
        policy_type="act",
        num_gpus=1,
        extra_train_args={
            "policy.n_action_steps": 30,
        },
    ),
    "diffusion": PolicySpec(
        name="diffusion",
        policy_type="diffusion",
        num_gpus=1,
        extra_train_args={
            "policy.horizon": 32,
            "policy.n_action_steps": 30,
            "policy.n_obs_steps": 2,
        },
    ),
    "smolvla": PolicySpec(
        name="smolvla",
        policy_type="smolvla",
        policy_path="lerobot/smolvla_base",
        num_gpus=8,
        extra_train_args={
            "policy.n_action_steps": 30,
            "policy.load_vlm_weights": True,
            "policy.freeze_vision_encoder": False,
            "policy.train_expert_only": False,
            "policy.scheduler_decay_steps": DEFAULT_STEPS,
            "policy.empty_cameras": 1,
        },
    ),
    "xvla": PolicySpec(
        name="xvla",
        policy_type="xvla",
        policy_path="lerobot/xvla-widowx",
        num_gpus=4,
        extra_train_args={
            "policy.n_action_steps": 32,
            "policy.scheduler_decay_steps": DEFAULT_STEPS,
            "policy.empty_cameras": 1,
        },
    ),
    "multi_task_dit": PolicySpec(
        name="multi_task_dit",
        policy_type="multi_task_dit",
        num_gpus=1,
        extra_train_args={
            "policy.horizon": 32,
            "policy.n_action_steps": 30,
        },
    ),
}


def normalize_repo_id(hub_org: str, repo_or_id: str) -> str:
    return repo_or_id if "/" in repo_or_id else f"{hub_org}/{repo_or_id}"


def get_requested_names(
    requested: list[str] | None,
    available: dict[str, Any],
    *,
    kind: str,
) -> list[str]:
    if not requested:
        return list(available)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown {kind}: {', '.join(unknown)}. Available: {', '.join(available)}")
    return requested


def compute_gradient_accumulation_steps(
    *,
    effective_batch_size: int,
    num_gpus: int,
    microbatch_per_gpu: int,
) -> int:
    per_step_batch = num_gpus * microbatch_per_gpu
    if effective_batch_size % per_step_batch != 0:
        raise ValueError(
            f"Cannot reach effective batch {effective_batch_size} with {num_gpus=} and "
            f"{microbatch_per_gpu=}."
        )
    return effective_batch_size // per_step_batch


def make_run_slug() -> str:
    return utc_timestamp_slug()


def shell_value(value: Any) -> str:
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True)
    else:
        value = str(value)
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )
    return f'"{escaped}"'


def format_cli_args(args: dict[str, Any]) -> str:
    lines = []
    for key, value in args.items():
        lines.append(f"  --{key}={shell_value(value)}")
    return " \\\n".join(lines)


def build_train_args(
    *,
    benchmark: BenchmarkSpec,
    policy: PolicySpec,
    train_dir: str,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "dataset.repo_id": benchmark.dataset_repo_id,
        "output_dir": train_dir,
        "steps": benchmark.train_steps,
        "batch_size": policy.microbatch_per_gpu,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "eval_freq": 0,
        "save_freq": benchmark.train_steps,
        "save_checkpoint": True,
        "log_freq": 100,
        "wandb.enable": False,
        "policy.push_to_hub": False,
        "policy.device": "cuda",
    }
    if policy.policy_path:
        args["policy.path"] = policy.policy_path
    else:
        args["policy.type"] = policy.policy_type
    args.update(benchmark.train_extra_args)
    args.update(policy.extra_train_args)
    return args


def build_eval_args(
    *,
    benchmark: BenchmarkSpec,
    policy: PolicySpec,
    checkpoint_path: str,
    eval_dir: str,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "policy.path": checkpoint_path,
        "env.type": benchmark.eval_env_type,
        "env.task": benchmark.eval_task,
        "eval.n_episodes": benchmark.eval_n_episodes,
        "output_dir": eval_dir,
    }
    args.update(benchmark.eval_extra_args)
    args.update(policy.extra_eval_args)
    return args


def plan_jobs(
    *,
    output_dir: Path,
    hub_org: str,
    results_repo: str,
    policies: list[str],
    benchmarks: list[str],
) -> list[PlannedJob]:
    _ = hub_org
    _ = results_repo
    scripts_dir = output_dir / "slurm"
    jobs: list[PlannedJob] = []
    for benchmark_name in benchmarks:
        benchmark = BENCHMARKS[benchmark_name]
        for policy_name in policies:
            policy = POLICIES[policy_name]
            num_gpus = max(MIN_GPUS, min(policy.num_gpus, MAX_GPUS))
            run_rel = f"runs/{benchmark_name}/{policy_name}/{make_run_slug()}"
            run_root = f"/benchmark-output/{run_rel}"
            gradient_accumulation_steps = compute_gradient_accumulation_steps(
                effective_batch_size=benchmark.effective_batch_size,
                num_gpus=num_gpus,
                microbatch_per_gpu=policy.microbatch_per_gpu,
            )
            train_dir = f"{run_root}/train"
            checkpoint_path = f"{train_dir}/checkpoints/{benchmark.train_steps:06d}/pretrained_model"
            eval_dir = f"{run_root}/eval"
            train_args = build_train_args(
                benchmark=benchmark,
                policy=policy,
                train_dir=train_dir,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            eval_args = build_eval_args(
                benchmark=benchmark,
                policy=policy,
                checkpoint_path=checkpoint_path,
                eval_dir=eval_dir,
            )
            tokenizer_args = None
            if policy.needs_tokenizer:
                tokenizer_repo_id = f"{hub_org}/{policy_name}-{benchmark_name}-tokenizer"
                tokenizer_args = {
                    "repo_id": benchmark.dataset_repo_id,
                    "output_dir": f"{run_root}/tokenizer",
                    "hub_repo_id": tokenizer_repo_id,
                    **policy.tokenizer_args,
                }
                train_args["policy.action_tokenizer_name"] = tokenizer_repo_id
            script_path = str(scripts_dir / f"{benchmark_name}__{policy_name}.sbatch")
            jobs.append(
                PlannedJob(
                    benchmark=benchmark_name,
                    policy=policy_name,
                    run_rel=run_rel,
                    num_gpus=num_gpus,
                    microbatch_per_gpu=policy.microbatch_per_gpu,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    effective_batch_size=benchmark.effective_batch_size,
                    docker_image=benchmark.docker_image,
                    train_args=train_args,
                    eval_args=eval_args,
                    tokenizer_args=tokenizer_args,
                    script_path=script_path,
                )
            )
    return jobs


def render_sbatch_script(
    *,
    job: PlannedJob,
    output_dir: Path,
    results_repo_id: str,
    git_commit: str,
) -> str:
    host_output_dir = output_dir.resolve()
    run_root = f"/benchmark-output/{job.run_rel}"
    host_run_root = host_output_dir / job.run_rel
    cpus_per_task = max(DEFAULT_CPUS_PER_GPU, DEFAULT_CPUS_PER_GPU * job.num_gpus)
    mem_gb = max(DEFAULT_MEMORY_PER_GPU_GB, DEFAULT_MEMORY_PER_GPU_GB * job.num_gpus)
    gpu_ids_expr = "${GPU_IDS}"
    train_cli = format_cli_args(job.train_args)
    eval_cli = format_cli_args(job.eval_args)
    tokenizer_command = ""
    if job.tokenizer_args:
        tokenizer_cli = format_cli_args(job.tokenizer_args)
        tokenizer_command = f"""
docker run --rm --gpus all \\
  --shm-size=16g \\
  -e CUDA_VISIBLE_DEVICES={gpu_ids_expr} \\
  -e HF_TOKEN="${{HF_TOKEN:-}}" \\
  -e HF_USER_TOKEN="${{HF_TOKEN:-}}" \\
  -e HF_HOME=/tmp/hf \\
  -v "{host_output_dir}:/benchmark-output" \\
  -w /lerobot \\
  "{job.docker_image}" \\
  bash -lc '
    set -euo pipefail
    if [[ -n "${{HF_TOKEN:-}}" ]]; then
      hf auth login --token "${{HF_TOKEN}}" --add-to-git-credential 2>/dev/null || true
    fi
    lerobot-train-tokenizer \\
{tokenizer_cli}
  '
"""
    return f"""#!/bin/bash
#SBATCH --job-name=bench-{job.benchmark}-{job.policy}
#SBATCH --gres=gpu:{job.num_gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem_gb}G
#SBATCH --output={output_dir.resolve()}/logs/{job.benchmark}__{job.policy}__%j.out
#SBATCH --error={output_dir.resolve()}/logs/{job.benchmark}__{job.policy}__%j.err

set -euo pipefail

HF_TOKEN="${{HF_TOKEN:-${{HF_USER_TOKEN:-}}}}"
GPU_IDS="$(seq -s, 0 $(({job.num_gpus} - 1)))"
RUN_ROOT="{run_root}"

mkdir -p "{host_output_dir}/logs"
mkdir -p "{host_run_root.parent}"

{tokenizer_command}

TRAIN_START="$(date +%s)"
docker run --rm --gpus all \\
  --shm-size=16g \\
  -e CUDA_VISIBLE_DEVICES="${{GPU_IDS}}" \\
  -e HF_TOKEN="${{HF_TOKEN:-}}" \\
  -e HF_USER_TOKEN="${{HF_TOKEN:-}}" \\
  -e HF_HOME=/tmp/hf \\
  -v "{host_output_dir}:/benchmark-output" \\
  -w /lerobot \\
  "{job.docker_image}" \\
  bash -lc '
    set -euo pipefail
    if [[ -n "${{HF_TOKEN:-}}" ]]; then
      hf auth login --token "${{HF_TOKEN}}" --add-to-git-credential 2>/dev/null || true
    fi
    accelerate launch --num_processes={job.num_gpus} $(which lerobot-train) \\
{train_cli}
  '
TRAIN_END="$(date +%s)"

EVAL_START="$(date +%s)"
docker run --rm --gpus all \\
  --shm-size=16g \\
  -e CUDA_VISIBLE_DEVICES="${{GPU_IDS}}" \\
  -e HF_TOKEN="${{HF_TOKEN:-}}" \\
  -e HF_USER_TOKEN="${{HF_TOKEN:-}}" \\
  -e HF_HOME=/tmp/hf \\
  -v "{host_output_dir}:/benchmark-output" \\
  -w /lerobot \\
  "{job.docker_image}" \\
  bash -lc '
    set -euo pipefail
    if [[ -n "${{HF_TOKEN:-}}" ]]; then
      hf auth login --token "${{HF_TOKEN}}" --add-to-git-credential 2>/dev/null || true
    fi
    lerobot-eval \\
{eval_cli}
  '
EVAL_END="$(date +%s)"
TRAIN_WALL_TIME_S="$((TRAIN_END - TRAIN_START))"
EVAL_WALL_TIME_S="$((EVAL_END - EVAL_START))"

docker run --rm --gpus all \\
  --shm-size=16g \\
  -e CUDA_VISIBLE_DEVICES="${{GPU_IDS}}" \\
  -e HF_TOKEN="${{HF_TOKEN:-}}" \\
  -e HF_USER_TOKEN="${{HF_TOKEN:-}}" \\
  -e HF_HOME=/tmp/hf \\
  -e RUN_ROOT="${{RUN_ROOT}}" \\
  -e TRAIN_WALL_TIME_S="${{TRAIN_WALL_TIME_S}}" \\
  -e EVAL_WALL_TIME_S="${{EVAL_WALL_TIME_S}}" \\
  -v "{host_output_dir}:/benchmark-output" \\
  -w /lerobot \\
  "{job.docker_image}" \\
  bash -lc '
    set -euo pipefail
    if [[ -n "${{HF_TOKEN:-}}" ]]; then
      hf auth login --token "${{HF_TOKEN}}" --add-to-git-credential 2>/dev/null || true
    fi
    uv run python benchmarks/publish_benchmark_result.py \\
      --benchmark={job.benchmark} \\
      --policy={job.policy} \\
      --run_root="${{RUN_ROOT}}" \\
      --results_repo={results_repo_id} \\
      --git_commit={git_commit} \\
      --num_gpus={job.num_gpus} \\
      --microbatch_per_gpu={job.microbatch_per_gpu} \\
      --gradient_accumulation_steps={job.gradient_accumulation_steps} \\
      --effective_batch_size={job.effective_batch_size} \\
      --train_wall_time_s="${{TRAIN_WALL_TIME_S}}" \\
      --eval_wall_time_s="${{EVAL_WALL_TIME_S}}" \\
      --slurm_job_id="${{SLURM_JOB_ID:-}}" \\
      --docker_image={job.docker_image}
  '
"""


def write_manifest(
    *,
    output_dir: Path,
    jobs: list[PlannedJob],
    git_commit: str,
    hub_org: str,
    results_repo: str,
) -> Path:
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "hub_org": hub_org,
        "results_repo": results_repo,
        "jobs": [asdict(job) for job in jobs],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", nargs="*", default=None)
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--hub_org", required=True)
    parser.add_argument("--results_repo", required=True)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def get_git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "slurm").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)

    selected_policies = get_requested_names(args.policies, POLICIES, kind="policies")
    selected_benchmarks = get_requested_names(args.benchmarks, BENCHMARKS, kind="benchmarks")
    git_commit = get_git_commit()
    results_repo_id = normalize_repo_id(args.hub_org, args.results_repo)

    jobs = plan_jobs(
        output_dir=args.output_dir,
        hub_org=args.hub_org,
        results_repo=results_repo_id,
        policies=selected_policies,
        benchmarks=selected_benchmarks,
    )

    for job in jobs:
        script = render_sbatch_script(
            job=job,
            output_dir=args.output_dir,
            results_repo_id=results_repo_id,
            git_commit=git_commit,
        )
        script_path = Path(job.script_path)
        script_path.write_text(script)
        script_path.chmod(0o755)
        if args.submit:
            subprocess.run(["sbatch", str(script_path)], check=True)

    manifest_path = write_manifest(
        output_dir=args.output_dir,
        jobs=jobs,
        git_commit=git_commit,
        hub_org=args.hub_org,
        results_repo=results_repo_id,
    )
    print(f"Wrote {len(jobs)} benchmark jobs to {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
