#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""Functional parity check: LeRobot Robometer vs. upstream Robometer.

Runs the in-tree :class:`RobometerRewardModel` on the same frames + task that
upstream Robometer was run on, and compares per-frame progress / success
predictions against reference outputs saved by upstream's
``scripts/example_inference_local.py``.

Workflow:

1. In the upstream Robometer environment (where ``robometer`` is importable),
   run::

       python third_party/robometer/scripts/example_inference_local.py \\
           --model-path robometer/Robometer-4B \\
           --video /path/to/episode.mp4 \\
           --task "Open the drawer" \\
           --fps 1.0 \\
           --out /tmp/robometer_upstream.npy

   This produces:
   - ``/tmp/robometer_upstream.npy``               (progress predictions)
   - ``/tmp/robometer_upstream_success_probs.npy`` (success probabilities)

2. Extract the exact same frames the upstream script used, save as ``.npz``::

       # quick helper: extract frames at the same fps and save as .npz
       python -c "
       from third_party.robometer.scripts.example_inference_local import load_frames_input
       import numpy as np
       frames = load_frames_input('/path/to/episode.mp4', fps=1.0, max_frames=512)
       np.savez('/tmp/robometer_frames.npz', frames=frames)
       "

3. In this LeRobot env, run this script::

       uv run python scripts/parity_robometer.py \\
           --frames /tmp/robometer_frames.npz \\
           --task "Open the drawer" \\
           --upstream-progress /tmp/robometer_upstream.npy \\
           --upstream-success  /tmp/robometer_upstream_success_probs.npy \\
           --lerobot-model     lilkm/robometer-4b
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel
from lerobot.rewards.robometer.modeling_robometer import decode_progress_outputs
from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep


def _load_frames(path: str) -> np.ndarray:
    """Load frames from .npy/.npz. Expects (T, H, W, C) uint8."""
    if path.endswith(".npy"):
        frames = np.load(path)
    elif path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as npz:
            frames = npz["frames"].copy() if "frames" in npz else next(iter(npz.values())).copy()
    else:
        raise ValueError(f"Frames must be .npy or .npz (got {path!r}).")

    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.ndim != 4:
        raise ValueError(f"Frames must be 4D (T,H,W,C); got shape {frames.shape}.")
    if frames.shape[-1] not in (1, 3):
        # Probably (T,C,H,W) — transpose
        if frames.shape[1] in (1, 3):
            frames = frames.transpose(0, 2, 3, 1)
        else:
            raise ValueError(f"Cannot interpret frame channel layout: {frames.shape}.")
    return frames


def _run_lerobot(
    frames: np.ndarray,
    task: str,
    model_path: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run LeRobot's Robometer on the given frames; return (progress, success)."""
    cfg = RobometerConfig(pretrained_path=model_path, device=device, max_frames=None)
    model = RobometerRewardModel.from_pretrained(model_path, config=cfg)

    encoder = RobometerEncoderProcessorStep(
        base_model_id=model.config.base_model_id,
        use_multi_image=model.config.use_multi_image,
        use_per_frame_progress_token=model.config.use_per_frame_progress_token,
        max_frames=None,
    )
    batch = encoder.encode_samples([(frames, task)])

    model_device = next(model.model.parameters()).device
    inputs = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in batch.items()}

    model.eval()
    with torch.no_grad():
        progress_logits, success_logits = model._compute_rbm_logits(inputs)

    decoded = decode_progress_outputs(
        progress_logits,
        success_logits,
        is_discrete_mode=model.config.use_discrete_progress,
    )
    progress = np.asarray(decoded["progress_pred"][0], dtype=np.float32)
    success = (
        np.asarray(decoded["success_probs"][0], dtype=np.float32)
        if decoded["success_probs"]
        else np.array([], dtype=np.float32)
    )
    return progress, success


def _compare(name: str, lerobot: np.ndarray, upstream: np.ndarray, atol: float, rtol: float) -> bool:
    print(f"\n=== {name} ===")
    if lerobot.shape != upstream.shape:
        print(f"shape mismatch: lerobot={lerobot.shape}  upstream={upstream.shape}")
        return False

    abs_diff = np.abs(lerobot - upstream)
    rel_diff = abs_diff / (np.abs(upstream) + 1e-12)
    print(f"shape        : {lerobot.shape}")
    print(f"max |Δ|      : {abs_diff.max():.3e}")
    print(f"mean |Δ|     : {abs_diff.mean():.3e}")
    print(f"max rel |Δ|  : {rel_diff.max():.3e}")
    print(f"lerobot[:5]  : {lerobot[:5]}")
    print(f"upstream[:5] : {upstream[:5]}")

    within_tol = bool(np.allclose(lerobot, upstream, atol=atol, rtol=rtol))
    print(f"allclose(atol={atol}, rtol={rtol}) -> {within_tol}")
    return within_tol


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--frames",
        required=True,
        help=".npy / .npz file with the exact frames upstream was run on (T,H,W,C uint8).",
    )
    parser.add_argument("--task", required=True, help="Task instruction string.")
    parser.add_argument(
        "--upstream-progress",
        required=True,
        help="Reference progress .npy saved by upstream example_inference_local.py.",
    )
    parser.add_argument(
        "--upstream-success",
        default=None,
        help="Optional reference success_probs .npy. If omitted, success comparison is skipped.",
    )
    parser.add_argument(
        "--lerobot-model",
        default="lilkm/robometer-4b",
        help="LeRobot-format Robometer Hub repo id or local path.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the LeRobot model (default: cuda if available).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for allclose (default: 1e-3; bf16 round-trip headroom).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for allclose (default: 1e-2).",
    )
    parser.add_argument(
        "--out-prefix",
        default="lerobot_robometer_outputs",
        help="Save the LeRobot outputs as <prefix>_progress.npy / <prefix>_success.npy.",
    )
    args = parser.parse_args()

    # 0. Sanity: confirm the LeRobot config is a RobometerConfig.
    cfg = RewardModelConfig.from_pretrained(args.lerobot_model)
    if not isinstance(cfg, RobometerConfig):
        print(f"ERROR: {args.lerobot_model!r} does not resolve to a RobometerConfig.", file=sys.stderr)
        return 2

    # 1. Load frames + task + upstream reference outputs.
    frames = _load_frames(args.frames)
    upstream_progress = np.load(args.upstream_progress).astype(np.float32)
    upstream_success = (
        np.load(args.upstream_success).astype(np.float32) if args.upstream_success is not None else None
    )

    print(f"Loaded {frames.shape[0]} frames at {frames.shape[1:]}, task={args.task!r}")
    print(f"LeRobot model: {args.lerobot_model}  device: {args.device}")

    # 2. Run LeRobot pipeline.
    progress, success = _run_lerobot(frames, args.task, args.lerobot_model, args.device)
    np.save(f"{args.out_prefix}_progress.npy", progress)
    if success.size > 0:
        np.save(f"{args.out_prefix}_success.npy", success)
    print(f"Saved LeRobot outputs to {args.out_prefix}_progress.npy / _success.npy")

    # 3. Compare to upstream references.
    progress_ok = _compare("progress", progress, upstream_progress, args.atol, args.rtol)
    if upstream_success is not None and success.size > 0:
        success_ok = _compare("success_probs", success, upstream_success, args.atol, args.rtol)
    else:
        success_ok = True
        print("\n(skipping success comparison — upstream success file not provided)")

    print()
    if progress_ok and success_ok:
        print("Parity check passed.")
        return 0
    print("Parity check FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
