#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train the RL-token bottleneck from frozen PI0 final-layer prefix tokens."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from lerobot.policies.pi0 import PI0Config, PI0Policy, make_pi0_pre_post_processors
from lerobot.policies.rl_token import RLTokenConfig, RLTokenModel, RLTokenStage1Trainer
from lerobot.utils.import_utils import require_package

TRAINER_STATE_NAME = "rl_token_trainer.pt"
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="lerobot/pi0_base")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--allow-short-run",
        action="store_true",
        help="Allow fewer than 2000 steps for smoke tests; paper runs use 2000-10000.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.steps > 10_000:
        raise ValueError("Stage 1 supports at most 10000 steps as specified by the paper")
    if args.steps < 2_000 and not args.allow_short_run:
        raise ValueError("Stage 1 requires 2000-10000 steps; use --allow-short-run only for smoke tests")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if args.log_freq <= 0 or args.save_freq <= 0:
        raise ValueError("log-freq and save-freq must be positive")


def _save_checkpoint(
    model: RLTokenModel,
    trainer: RLTokenStage1Trainer,
    output_dir: Path,
    *,
    numbered: bool,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{trainer.steps}" if numbered else output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    torch.save(trainer.state_dict(), checkpoint_dir / TRAINER_STATE_NAME)


def train(args: argparse.Namespace) -> RLTokenStage1Trainer:
    _validate_args(args)
    require_package("datasets", extra="training")
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.collate import lerobot_collate_fn

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; using CPU")
        requested_device = torch.device("cpu")

    dataset = LeRobotDataset(repo_id=args.dataset_repo_id, root=args.dataset_root)
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=requested_device.type == "cuda",
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    if len(dataloader) == 0:
        raise ValueError("the Stage 1 dataset is empty")

    pi0_config = PI0Config.from_pretrained(
        args.policy_path,
        local_files_only=args.local_files_only,
        device=str(requested_device),
    )
    pi0 = PI0Policy.from_pretrained(
        args.policy_path,
        config=pi0_config,
        local_files_only=args.local_files_only,
        strict=True,
    ).eval()
    if not pi0._pretrained_weights_loaded:  # noqa: SLF001
        raise RuntimeError(
            f"PI0 weights were not loaded from {args.policy_path!r}; "
            "refusing to train RL-token on random VLA features"
        )
    pi0.requires_grad_(False)
    preprocessor, _ = make_pi0_pre_post_processors(config=pi0_config, dataset_stats=dataset.meta.stats)

    trainer: RLTokenStage1Trainer | None = None
    iterator = iter(dataloader)
    while trainer is None or trainer.steps < args.steps:
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            raw_batch = next(iterator)
        batch = preprocessor(raw_batch)
        final_tokens, token_mask = pi0.encode_vla_tokens(batch)

        if trainer is None:
            if args.resume is not None:
                model = RLTokenModel.from_pretrained(args.resume, map_location=requested_device)
            else:
                model = RLTokenModel(
                    RLTokenConfig(
                        vla_dim=final_tokens.shape[-1],
                        token_dim=args.token_dim,
                        max_tokens=args.max_tokens,
                        encoder_layers=args.encoder_layers,
                        decoder_layers=args.decoder_layers,
                        num_heads=args.num_heads,
                    )
                ).to(requested_device)
            trainer = RLTokenStage1Trainer(
                model,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
            )
            if args.resume is not None:
                state = torch.load(
                    args.resume / TRAINER_STATE_NAME,
                    map_location=requested_device,
                    weights_only=True,
                )
                trainer.load_state_dict(state)

        if trainer.steps >= args.steps:
            break
        metrics = trainer.step(final_tokens.to(requested_device), token_mask.to(requested_device))
        if trainer.steps % args.log_freq == 0 or trainer.steps == 1:
            logger.info(
                "step=%d reconstruction_loss=%.6f grad_norm=%.4f",
                trainer.steps,
                metrics["reconstruction_loss"],
                metrics["token_grad_norm"],
            )
        if trainer.steps % args.save_freq == 0:
            _save_checkpoint(trainer.model, trainer, args.output_dir, numbered=True)

    _save_checkpoint(trainer.model, trainer, args.output_dir, numbered=False)
    return trainer


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
