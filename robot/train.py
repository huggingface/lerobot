#!/usr/bin/env python
"""
Train a manipulation policy (ACT or Diffusion) on collected OAK-D episodes.

Wraps LeRobot's training pipeline with defaults tuned for the SO-100 + OAK-D setup.

Usage::

    # Train ACT policy on collected episodes
    python -m robot.train --policy act --dataset-repo-id local/pick_red_cube

    # Train Diffusion Policy
    python -m robot.train --policy diffusion --dataset-repo-id local/pick_red_cube

    # Resume from checkpoint
    python -m robot.train --policy act --dataset-repo-id local/pick_red_cube --resume

    # Custom training params
    python -m robot.train --policy act --dataset-repo-id local/pick_red_cube \\
        --batch-size 16 --steps 200000 --lr 1e-4
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

import yaml

logger = logging.getLogger(__name__)


def build_train_command(
    policy: str,
    dataset_repo_id: str,
    output_dir: str = "./checkpoints",
    batch_size: int = 8,
    steps: int = 100_000,
    lr: float | None = None,
    resume: bool = False,
    num_workers: int = 4,
    save_freq: int = 20_000,
    eval_freq: int = 20_000,
    wandb_project: str = "",
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the lerobot-train CLI command.

    Args:
        policy: Policy type — "act" or "diffusion".
        dataset_repo_id: HuggingFace dataset repo ID.
        output_dir: Directory for checkpoints and logs.
        batch_size: Training batch size.
        steps: Total training steps.
        lr: Learning rate override (uses policy default if None).
        resume: Resume from latest checkpoint in output_dir.
        num_workers: Dataloader workers.
        save_freq: Save checkpoint every N steps.
        eval_freq: Evaluate every N steps.
        wandb_project: W&B project name (empty = disabled).
        extra_args: Additional CLI arguments to pass through.

    Returns:
        Command as list of strings.
    """
    policy_type = {
        "act": "act",
        "diffusion": "diffusion",
    }.get(policy.lower(), policy)

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--policy.type={policy_type}",
        f"--output_dir={output_dir}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--num_workers={num_workers}",
        f"--save_freq={save_freq}",
        f"--eval_freq={eval_freq}",
    ]

    if lr is not None:
        cmd.append(f"--optimizer.lr={lr}")

    if resume:
        cmd.append("--resume=true")

    if wandb_project:
        cmd.extend([
            f"--wandb.project={wandb_project}",
            "--wandb.enable=true",
        ])
    else:
        cmd.append("--wandb.enable=false")

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def train(args: argparse.Namespace) -> None:
    """Run training."""
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    lr_cfg = cfg.get("lerobot", {})

    policy = args.policy or lr_cfg.get("policy", "act")
    dataset_repo_id = args.dataset_repo_id or lr_cfg.get("dataset_repo_id", "")
    output_dir = args.output_dir or lr_cfg.get("checkpoint_path", "./checkpoints")

    if not dataset_repo_id:
        logger.error("--dataset-repo-id is required (or set lerobot.dataset_repo_id in config.yaml)")
        sys.exit(1)

    cmd = build_train_command(
        policy=policy,
        dataset_repo_id=dataset_repo_id,
        output_dir=output_dir,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        resume=args.resume,
        num_workers=args.num_workers,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        wandb_project=args.wandb_project,
        extra_args=args.extra,
    )

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Train manipulation policy")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--policy", default="", help="Policy type: act or diffusion")
    parser.add_argument("--dataset-repo-id", default="", help="Dataset repo ID")
    parser.add_argument("--output-dir", default="", help="Checkpoint output directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=20_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--wandb-project", default="", help="W&B project (empty = disabled)")
    parser.add_argument("extra", nargs="*", help="Extra args passed to lerobot-train")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    train(args)


if __name__ == "__main__":
    main()
