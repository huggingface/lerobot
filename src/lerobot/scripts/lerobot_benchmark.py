#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Benchmark runner: train and evaluate policies across simulation benchmarks.

Orchestrates per-benchmark training and evaluation using the existing
``lerobot-train`` and ``lerobot-eval`` CLI tools.

Typical usage::

    # Train SmolVLA on LIBERO-plus (4 GPUs, 50k steps):
    lerobot-benchmark train \\
        --benchmarks libero_plus \\
        --policy-path lerobot/smolvla_base \\
        --hub-user $HF_USER \\
        --num-gpus 4 --steps 50000

    # Evaluate the trained policies:
    lerobot-benchmark eval \\
        --benchmarks libero_plus \\
        --hub-user $HF_USER

    # Full pipeline (train → upload → eval) for multiple benchmarks:
    lerobot-benchmark all \\
        --benchmarks libero_plus,robocasa,robomme \\
        --policy-path lerobot/smolvla_base \\
        --hub-user $HF_USER \\
        --num-gpus 4 --steps 50000
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class BenchmarkEntry:
    """Training + evaluation settings for a single benchmark.

    When ``eval_tasks`` is set, evaluation runs once per task in the list
    (e.g. libero_spatial, libero_object, …).  ``env_task`` is still used as
    the task for mid-training evaluation during ``lerobot-train``.
    """

    dataset_repo_id: str
    env_type: str
    env_task: str
    eval_tasks: list[str] | None = None
    train_overrides: dict[str, str] = field(default_factory=dict)
    eval_overrides: dict[str, str] = field(default_factory=dict)


LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# Each benchmark maps a human-readable name to its dataset and eval env.
# ``dataset_repo_id`` can contain ``{hub_user}`` which is interpolated at
# runtime from ``--hub-user``.
BENCHMARK_REGISTRY: dict[str, BenchmarkEntry] = {
    "libero": BenchmarkEntry(
        dataset_repo_id="{hub_user}/libero",
        env_type="libero",
        env_task="libero_spatial",
        eval_tasks=LIBERO_SUITES,
    ),
    "libero_plus": BenchmarkEntry(
        dataset_repo_id="{hub_user}/libero_plus",
        env_type="libero_plus",
        env_task="libero_spatial",
        eval_tasks=LIBERO_SUITES,
    ),
    "metaworld": BenchmarkEntry(
        dataset_repo_id="{hub_user}/metaworld",
        env_type="metaworld",
        env_task="metaworld-push-v2",
    ),
    "robocasa": BenchmarkEntry(
        dataset_repo_id="{hub_user}/robocasa",
        env_type="robocasa",
        env_task="PickPlaceCounterToCabinet",
    ),
    "robomme": BenchmarkEntry(
        dataset_repo_id="{hub_user}/robomme",
        env_type="robomme",
        env_task="PickXtimes",
    ),
}


def _policy_repo_id(hub_user: str, policy_name: str, benchmark: str) -> str:
    return f"{hub_user}/{policy_name}_{benchmark}"


def _extra_keys(extra_args: list[str]) -> set[str]:
    """Extract ``--key`` prefixes from extra CLI args for override detection."""
    keys: set[str] = set()
    for arg in extra_args:
        if arg.startswith("--") and "=" in arg:
            keys.add(arg.split("=", 1)[0])
    return keys


def _build_train_cmd(
    benchmark: BenchmarkEntry,
    *,
    policy_path: str,
    hub_user: str,
    policy_name: str,
    benchmark_name: str,
    num_gpus: int,
    steps: int,
    batch_size: int,
    eval_freq: int,
    save_freq: int,
    wandb: bool,
    extra_args: list[str],
) -> list[str]:
    """Build the ``accelerate launch lerobot-train`` command list."""
    lerobot_train = shutil.which("lerobot-train")
    if lerobot_train is None:
        raise RuntimeError("lerobot-train not found on PATH. Is lerobot installed?")

    # Strip bare "--" separators that argparse may pass through
    cleaned_extra = [a for a in extra_args if a != "--"]
    overridden = _extra_keys(cleaned_extra)

    repo_id = _policy_repo_id(hub_user, policy_name, benchmark_name)
    dataset_id = benchmark.dataset_repo_id.format(hub_user=hub_user)

    defaults: list[tuple[str, str]] = [
        ("--policy.path", policy_path),
        ("--dataset.repo_id", dataset_id),
        ("--policy.repo_id", repo_id),
        ("--env.type", benchmark.env_type),
        ("--env.task", benchmark.env_task),
        ("--steps", str(steps)),
        ("--batch_size", str(batch_size)),
        ("--eval_freq", str(eval_freq)),
        ("--save_freq", str(save_freq)),
        ("--output_dir", f"outputs/train/{policy_name}_{benchmark_name}"),
        ("--job_name", f"{policy_name}_{benchmark_name}"),
        ("--policy.push_to_hub", "true"),
    ]
    if wandb:
        defaults.append(("--wandb.enable", "true"))
    for k, v in benchmark.train_overrides.items():
        defaults.append((f"--{k}", v))

    cmd: list[str] = [
        "accelerate", "launch",
        "--multi_gpu",
        f"--num_processes={num_gpus}",
        lerobot_train,
    ]
    for key, val in defaults:
        if key not in overridden:
            cmd.append(f"{key}={val}")
    cmd.extend(cleaned_extra)
    return cmd


def _build_eval_cmd(
    benchmark: BenchmarkEntry,
    *,
    hub_user: str,
    policy_name: str,
    benchmark_name: str,
    eval_task: str | None = None,
    n_episodes: int,
    batch_size_eval: int,
    extra_args: list[str],
) -> list[str]:
    """Build the ``lerobot-eval`` command list.

    ``eval_task`` overrides the benchmark's ``env_task`` so the same
    benchmark can be evaluated on multiple suites (e.g. LIBERO).
    """
    lerobot_eval = shutil.which("lerobot-eval")
    if lerobot_eval is None:
        raise RuntimeError("lerobot-eval not found on PATH. Is lerobot installed?")

    task = eval_task or benchmark.env_task
    repo_id = _policy_repo_id(hub_user, policy_name, benchmark_name)
    out_dir = _eval_output_dir(policy_name, benchmark_name, eval_task=task)

    cleaned_extra = [a for a in extra_args if a != "--"]
    overridden = _extra_keys(cleaned_extra)

    defaults: list[tuple[str, str]] = [
        ("--policy.path", repo_id),
        ("--env.type", benchmark.env_type),
        ("--env.task", task),
        ("--eval.n_episodes", str(n_episodes)),
        ("--eval.batch_size", str(batch_size_eval)),
        ("--output_dir", out_dir),
        ("--policy.device", "cuda"),
    ]
    for k, v in benchmark.eval_overrides.items():
        defaults.append((f"--{k}", v))

    cmd: list[str] = [lerobot_eval]
    for key, val in defaults:
        if key not in overridden:
            cmd.append(f"{key}={val}")
    cmd.extend(cleaned_extra)
    return cmd


def _eval_output_dir(policy_name: str, benchmark_name: str, eval_task: str | None = None) -> Path:
    if eval_task:
        return Path(f"outputs/eval/{policy_name}_{benchmark_name}/{eval_task}")
    return Path(f"outputs/eval/{policy_name}_{benchmark_name}")


def _run(cmd: list[str], *, dry_run: bool) -> None:
    log.info("Command: %s", " \\\n    ".join(cmd))
    if dry_run:
        log.info("[dry-run] Skipping execution.")
        return
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error("Command failed with exit code %d", result.returncode)
        sys.exit(result.returncode)


def _push_eval_to_hub(
    *,
    hub_user: str,
    policy_name: str,
    benchmark_name: str,
    eval_task: str | None = None,
    dry_run: bool,
) -> None:
    """Upload eval results (metrics + videos) to the policy repo on the Hub."""
    from huggingface_hub import HfApi

    repo_id = _policy_repo_id(hub_user, policy_name, benchmark_name)
    local_dir = _eval_output_dir(policy_name, benchmark_name, eval_task=eval_task)
    hub_path = f"eval/{eval_task}" if eval_task else f"eval/{benchmark_name}"

    if not local_dir.exists():
        log.warning("Eval output dir %s does not exist, skipping hub upload.", local_dir)
        return

    log.info("Uploading eval results from %s to %s (path_in_repo=%s)", local_dir, repo_id, hub_path)
    if dry_run:
        log.info("[dry-run] Skipping upload.")
        return

    api = HfApi()
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        path_in_repo=hub_path,
        repo_type="model",
        commit_message=f"Upload eval results for {eval_task or benchmark_name}",
    )


def _resolve_benchmarks(names: str) -> list[tuple[str, BenchmarkEntry]]:
    out = []
    for name in names.split(","):
        name = name.strip()
        if name not in BENCHMARK_REGISTRY:
            available = ", ".join(BENCHMARK_REGISTRY)
            raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
        out.append((name, BENCHMARK_REGISTRY[name]))
    return out


def cmd_train(args: argparse.Namespace) -> None:
    benchmarks = _resolve_benchmarks(args.benchmarks)
    for bname, bentry in benchmarks:
        log.info("=== Training on benchmark: %s ===", bname)
        cmd = _build_train_cmd(
            bentry,
            policy_path=args.policy_path,
            hub_user=args.hub_user,
            policy_name=args.policy_name,
            benchmark_name=bname,
            num_gpus=args.num_gpus,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            wandb=args.wandb,
            extra_args=args.extra,
        )
        _run(cmd, dry_run=args.dry_run)


def _run_eval_for_benchmark(
    bname: str,
    bentry: BenchmarkEntry,
    args: argparse.Namespace,
) -> None:
    """Run evaluation for a single benchmark, iterating over all its eval_tasks."""
    tasks = bentry.eval_tasks or [bentry.env_task]
    for task in tasks:
        log.info("=== Evaluating %s / %s ===", bname, task)
        cmd = _build_eval_cmd(
            bentry,
            hub_user=args.hub_user,
            policy_name=args.policy_name,
            benchmark_name=bname,
            eval_task=task if bentry.eval_tasks else None,
            n_episodes=args.n_episodes,
            batch_size_eval=args.batch_size_eval,
            extra_args=args.extra,
        )
        _run(cmd, dry_run=args.dry_run)
        if args.push_eval_to_hub:
            _push_eval_to_hub(
                hub_user=args.hub_user,
                policy_name=args.policy_name,
                benchmark_name=bname,
                eval_task=task if bentry.eval_tasks else None,
                dry_run=args.dry_run,
            )


def cmd_eval(args: argparse.Namespace) -> None:
    benchmarks = _resolve_benchmarks(args.benchmarks)
    for bname, bentry in benchmarks:
        _run_eval_for_benchmark(bname, bentry, args)


def cmd_all(args: argparse.Namespace) -> None:
    """Train on each benchmark, then evaluate each."""
    benchmarks = _resolve_benchmarks(args.benchmarks)

    log.info("Phase 1: Training on %d benchmark(s)", len(benchmarks))
    for bname, bentry in benchmarks:
        log.info("=== Training on benchmark: %s ===", bname)
        cmd = _build_train_cmd(
            bentry,
            policy_path=args.policy_path,
            hub_user=args.hub_user,
            policy_name=args.policy_name,
            benchmark_name=bname,
            num_gpus=args.num_gpus,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            wandb=args.wandb,
            extra_args=args.extra,
        )
        _run(cmd, dry_run=args.dry_run)

    log.info("Phase 2: Evaluating %d benchmark(s)", len(benchmarks))
    for bname, bentry in benchmarks:
        _run_eval_for_benchmark(bname, bentry, args)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--benchmarks", required=True,
        help="Comma-separated benchmark names (e.g. libero_plus,robocasa,robomme).",
    )
    p.add_argument("--hub-user", required=True, help="HuggingFace Hub username.")
    p.add_argument(
        "--policy-name", default="smolvla",
        help="Short policy name used in repo IDs and output dirs (default: smolvla).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--policy-path", default="lerobot/smolvla_base", help="Pretrained policy path.")
    p.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs.")
    p.add_argument("--steps", type=int, default=50_000, help="Total training steps.")
    p.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size.")
    p.add_argument("--eval-freq", type=int, default=10_000, help="Eval every N steps (0 to disable).")
    p.add_argument("--save-freq", type=int, default=10_000, help="Save checkpoint every N steps.")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")


def _add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-episodes", type=int, default=50, help="Number of eval episodes.")
    p.add_argument("--batch-size-eval", type=int, default=10, help="Eval batch size (parallel envs).")
    p.add_argument(
        "--push-eval-to-hub", action="store_true",
        help="Upload eval results (metrics + videos) to the policy repo on the Hub.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot-benchmark",
        description="Train and evaluate policies across simulation benchmarks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train a policy on each selected benchmark.")
    _add_common_args(p_train)
    _add_train_args(p_train)
    p_train.set_defaults(func=cmd_train)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate trained policies on each benchmark.")
    _add_common_args(p_eval)
    _add_eval_args(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    # all (train + eval)
    p_all = sub.add_parser("all", help="Train then evaluate on each benchmark.")
    _add_common_args(p_all)
    _add_train_args(p_all)
    _add_eval_args(p_all)
    p_all.set_defaults(func=cmd_all)

    # list
    p_list = sub.add_parser("list", help="List available benchmarks.")
    p_list.set_defaults(func=lambda _args: _list_benchmarks())

    return parser


def _list_benchmarks() -> None:
    print("Available benchmarks:\n")
    for name, entry in BENCHMARK_REGISTRY.items():
        print(f"  {name}")
        print(f"    dataset:  {entry.dataset_repo_id}")
        print(f"    env:      {entry.env_type}")
        if entry.eval_tasks:
            print(f"    eval on:  {', '.join(entry.eval_tasks)}")
        else:
            print(f"    eval on:  {entry.env_task}")
        print()


def main() -> None:
    parser = build_parser()
    args, extra = parser.parse_known_args()
    args.extra = extra
    args.func(args)


if __name__ == "__main__":
    main()
