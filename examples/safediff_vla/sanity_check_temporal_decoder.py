#!/usr/bin/env python
"""Pre-flight checks for the `temporal_decoder` / `temporal_decoder_subgoal` architectures,
against the *real* `lerobot/smolvla_vlabench` backbone and a handful of real
`lerobot/vlabench_unified` frames — run this before any real (20k+ step) training, not after.

Checks (see SafeDiff-VLA redesign spec, section 11):
  1. Shape test: one forward pass, every major tensor's shape logged and asserted.
  2. Gradient test: backbone has no grad, decoder does.
  3. Overfit test: repeated optimizer steps on the same tiny batch must drive the loss down
     substantially. If a decoder can't even overfit a handful of samples, it has no business
     being pointed at a real training run.
  4. Action-distribution check: target vs. predicted action mean/std/min/max, per-step
     difference, and acceleration magnitude, before and after the overfit loop -- catches
     normalization breakage that a scalar loss alone can hide.

Usage:
    uv run python examples/safediff_vla/sanity_check_temporal_decoder.py
    uv run python examples/safediff_vla/sanity_check_temporal_decoder.py --architecture temporal_decoder_subgoal
"""

import argparse
import logging
import sys

import torch

from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig
from lerobot.processor.rename_processor import rename_batch_keys

# `force=True`: some already-imported lerobot module configures the root logger (or raises its
# level) before this runs, which silently makes a plain `basicConfig()` call here a no-op.
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
SUBGOAL_LABELS_PATH = "outputs/data/vlabench_subgoal_labels/labels.parquet"


def describe(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        logger.info("%-28s None", name)
        return
    logger.info("%-28s shape=%s dtype=%s device=%s", name, tuple(tensor.shape), tensor.dtype, tensor.device)


def action_distribution_stats(label: str, actions: torch.Tensor) -> None:
    velocity = actions[:, 1:] - actions[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    logger.info(
        "%-16s mean=%.4f std=%.4f min=%.4f max=%.4f |da/dt|=%.4f |d2a/dt2|=%.4f",
        label,
        actions.mean().item(),
        actions.std().item(),
        actions.min().item(),
        actions.max().item(),
        velocity.abs().mean().item(),
        acceleration.abs().mean().item(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--architecture",
        default="temporal_decoder",
        choices=["temporal_decoder", "temporal_decoder_subgoal"],
    )
    parser.add_argument("--backbone-name", default="lerobot/smolvla_vlabench")
    parser.add_argument("--repo-id", default="lerobot/vlabench_unified")
    parser.add_argument("--episodes", type=int, nargs="+", default=[3, 6, 7, 8])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overfit-steps", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    use_subgoal = args.architecture == "temporal_decoder_subgoal"
    config = SafeDiffVLAConfig(
        architecture=args.architecture,
        backbone_name=args.backbone_name,
        device=args.device,
        freeze_backbone=True,
        subgoal_labels_path=SUBGOAL_LABELS_PATH if use_subgoal else None,
    )

    logger.info("=== building dataset (episodes=%s) ===", args.episodes)
    LeRobotDatasetMetadata(args.repo_id)  # populates the local HF cache before make_dataset needs it

    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=args.repo_id, episodes=args.episodes),
        policy=config,
        rename_map=RENAME_MAP,
        batch_size=args.batch_size,
    )
    dataset = make_dataset(cfg)

    logger.info(
        "=== building policy (architecture=%s, backbone=%s) ===", args.architecture, args.backbone_name
    )
    policy = make_policy(cfg=config, ds_meta=dataset.meta, rename_map=RENAME_MAP)
    policy = policy.to(args.device)
    policy.train()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    raw_batch = next(iter(loader))
    # Mirrors `lerobot_train.py`'s `_preprocess_dataset_batch`: for a *fresh* (non-resumed) run,
    # renaming happens directly on the raw batch before the processor pipeline ever sees it, not
    # via a "rename_observations_processor" override (that override path only fires when
    # resuming from a `pretrained_path`).
    for cam_key in dataset.meta.camera_keys:
        if cam_key in raw_batch and raw_batch[cam_key].dtype == torch.uint8:
            raw_batch[cam_key] = raw_batch[cam_key].to(dtype=torch.float32) / 255.0
    raw_batch = rename_batch_keys(raw_batch, RENAME_MAP)
    batch = preprocessor(raw_batch)

    # ---- 1. shape test ------------------------------------------------------------------
    logger.info("=== 1. shape test ===")
    latent_tokens, latent_pad_mask = policy._encode_multimodal_latent(batch)
    current_state = policy._current_state(batch)
    describe("VLM latent tokens", latent_tokens)
    describe("latent pad mask", latent_pad_mask)
    describe("current_state", current_state)
    describe("target actions", batch["action"])
    if use_subgoal:
        pooled = policy.latent_pool_projection(policy._pooled_latent(latent_tokens, latent_pad_mask))
        predicted_subgoal = policy.subgoal_state_predictor(pooled, current_state)
        describe("predicted subgoal state", predicted_subgoal)
    pred_actions, metrics = policy.plan_action_chunk(batch)
    describe("predicted actions", pred_actions)
    assert pred_actions.shape == batch["action"].shape[:1] + (
        config.action_horizon,
        config.action_feature.shape[0],
    ), (
        f"predicted_actions shape {pred_actions.shape} does not match "
        f"(B, {config.action_horizon}, {config.action_feature.shape[0]})"
    )
    logger.info("shape test OK")

    # ---- 2. gradient test -----------------------------------------------------------------
    logger.info("=== 2. gradient test ===")
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
    loss, train_metrics = policy(batch)
    logger.info("initial loss=%.4f metrics=%s", loss.item(), train_metrics)
    optimizer.zero_grad()
    loss.backward()
    backbone_has_grad = any(p.grad is not None for p in policy.backbone.parameters())
    decoder_has_grad = any(p.grad is not None for p in policy.decoder.parameters())
    logger.info(
        "backbone_has_grad=%s (expected False) decoder_has_grad=%s (expected True)",
        backbone_has_grad,
        decoder_has_grad,
    )
    assert not backbone_has_grad, "backbone should be frozen -- got gradients!"
    assert decoder_has_grad, "decoder should have gradients but doesn't!"
    optimizer.step()
    logger.info("gradient test OK")

    # ---- 3 & 4. overfit test + action distribution -----------------------------------------
    logger.info("=== 3. overfit test (%d steps on this one tiny batch) ===", args.overfit_steps)
    action_distribution_stats("target", batch["action"])
    action_distribution_stats("pred (before)", pred_actions.detach())

    losses = []
    for step in range(args.overfit_steps):
        loss, train_metrics = policy(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % max(1, args.overfit_steps // 10) == 0:
            logger.info("step %4d loss=%.4f metrics=%s", step, loss.item(), train_metrics)

    with torch.no_grad():
        pred_actions_after, _ = policy.plan_action_chunk(batch)
    action_distribution_stats("pred (after)", pred_actions_after)

    logger.info(
        "loss[0]=%.4f -> loss[-1]=%.4f (ratio=%.3f)", losses[0], losses[-1], losses[-1] / max(losses[0], 1e-8)
    )
    assert losses[-1] < losses[0] * 0.5, (
        f"loss barely moved ({losses[0]:.4f} -> {losses[-1]:.4f}) -- decoder can't even overfit "
        "a handful of samples; do NOT proceed to full training until this passes."
    )
    logger.info("overfit test OK -- safe to proceed to a real training run.")


if __name__ == "__main__":
    main()
