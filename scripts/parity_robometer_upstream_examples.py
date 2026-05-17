#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""Run LeRobot Robometer parity against upstream Robometer's bundled examples.

Upstream Robometer ships three reference videos with their pre-computed
progress / success outputs at
``third_party/robometer/scripts/example_videos/``::

    soar_put_green_stick_in_brown_bowl.mp4
        + soar_put_green_stick_in_brown_bowl_rewards.npy            (progress)
        + soar_put_green_stick_in_brown_bowl_rewards_success_probs.npy (success)
    berkeley_rpt_stack_cup.mp4
        + berkeley_rpt_stack_cup_rewards.npy
        + berkeley_rpt_stack_cup_rewards_success_probs.npy
    jaco_play_pick_up_green_cup.mp4
        + pick_up_green_cup_rewards.npy
        + pick_up_green_cup_rewards_success_probs.npy

This script:
1. Decodes each video at upstream's sampling fps using ``av`` (PyAV), with the
   same linspace-over-total-frames logic as upstream's ``extract_frames``.
2. Runs the LeRobot ``RobometerRewardModel`` on those frames + the task from
   upstream's README.
3. Compares per-frame progress / success to the pre-saved upstream outputs.

This means you do **not** need to install upstream Robometer to confirm parity.

Run::

    uv run python scripts/parity_robometer_upstream_examples.py \\
        --lerobot-model lilkm/robometer-4b \\
        --device cuda \\
        --fps 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import numpy as np
import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel
from lerobot.rewards.robometer.modeling_robometer import decode_progress_outputs
from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep

EXAMPLES = [
    {
        "name": "soar_put_green_stick_in_brown_bowl",
        "video": "soar_put_green_stick_in_brown_bowl.mp4",
        "task": "Put green stick in brown bowl",
        "progress_npy": "soar_put_green_stick_in_brown_bowl_rewards.npy",
        "success_npy": "soar_put_green_stick_in_brown_bowl_rewards_success_probs.npy",
    },
    {
        "name": "berkeley_rpt_stack_cup",
        "video": "berkeley_rpt_stack_cup.mp4",
        "task": "Pick up the yellow cup and stack it on the other cup",
        "progress_npy": "berkeley_rpt_stack_cup_rewards.npy",
        "success_npy": "berkeley_rpt_stack_cup_rewards_success_probs.npy",
    },
    {
        "name": "jaco_play_pick_up_green_cup",
        "video": "jaco_play_pick_up_green_cup.mp4",
        "task": "Pick up the green cup",
        "progress_npy": "pick_up_green_cup_rewards.npy",
        "success_npy": "pick_up_green_cup_rewards_success_probs.npy",
    },
]


def _extract_frames_av(video_path: Path, fps: float) -> np.ndarray:
    """Mirror upstream's ``extract_frames`` sampling logic using PyAV.

    Upstream uses ``decord`` to read all frames, then samples
    ``np.linspace(0, total_frames - 1, desired_frames, dtype=int)`` where
    ``desired_frames = round(total_frames * (fps / native_fps))``. We do the
    same here so the per-frame outputs are directly comparable.
    """
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    native_fps = float(stream.average_rate) if stream.average_rate else float(stream.guessed_rate or 30.0)

    rgb_frames: list[np.ndarray] = []
    for frame in container.decode(stream):
        rgb_frames.append(frame.to_ndarray(format="rgb24"))
    container.close()

    total_frames = len(rgb_frames)
    if total_frames == 0:
        raise RuntimeError(f"No decodable frames in {video_path}.")

    desired_frames = max(1, int(round(total_frames * (fps / max(native_fps, 1e-6)))))
    desired_frames = min(desired_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int)
    return np.stack([rgb_frames[i] for i in indices])


def _run_lerobot(
    model: RobometerRewardModel,
    encoder: RobometerEncoderProcessorStep,
    frames: np.ndarray,
    task: str,
) -> tuple[np.ndarray, np.ndarray]:
    batch = encoder.encode_samples([(frames, task)])
    device = next(model.model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    model.eval()
    with torch.no_grad():
        progress_logits, success_logits = model._compute_rbm_logits(inputs)
    decoded = decode_progress_outputs(
        progress_logits, success_logits, is_discrete_mode=model.config.use_discrete_progress
    )
    progress = np.asarray(decoded["progress_pred"][0], dtype=np.float32)
    success = (
        np.asarray(decoded["success_probs"][0], dtype=np.float32)
        if decoded["success_probs"]
        else np.array([], dtype=np.float32)
    )
    return progress, success


def _compare(name: str, lerobot: np.ndarray, upstream: np.ndarray, atol: float, rtol: float) -> bool:
    if lerobot.shape != upstream.shape:
        print(f"  {name}: shape mismatch  lerobot={lerobot.shape}  upstream={upstream.shape}")
        return False
    abs_diff = np.abs(lerobot - upstream)
    print(f"  {name:16s} shape={lerobot.shape}  max|Δ|={abs_diff.max():.3e}  mean|Δ|={abs_diff.mean():.3e}")
    return bool(np.allclose(lerobot, upstream, atol=atol, rtol=rtol))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("third_party/robometer/scripts/example_videos"),
        help="Directory containing the upstream Robometer example mp4s + .npy outputs.",
    )
    parser.add_argument(
        "--lerobot-model",
        default="lilkm/robometer-4b",
        help="LeRobot-format Robometer Hub repo id or local path.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the LeRobot model.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="Sampling fps (default: 3, matching the upstream README).",
    )
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    examples_dir = args.examples_dir.resolve()
    if not examples_dir.is_dir():
        print(f"ERROR: examples dir {examples_dir} does not exist.", file=sys.stderr)
        return 2

    # Sanity-check the LeRobot config is a RobometerConfig before loading weights.
    cfg = RewardModelConfig.from_pretrained(args.lerobot_model)
    if not isinstance(cfg, RobometerConfig):
        print(f"ERROR: {args.lerobot_model!r} did not resolve to a RobometerConfig.", file=sys.stderr)
        return 2

    print(f"Loading LeRobot Robometer from {args.lerobot_model} on {args.device}...")
    cfg.pretrained_path = args.lerobot_model
    cfg.device = args.device
    model = RobometerRewardModel.from_pretrained(args.lerobot_model, config=cfg)
    encoder = RobometerEncoderProcessorStep(
        base_model_id=model.config.base_model_id,
        use_multi_image=model.config.use_multi_image,
        use_per_frame_progress_token=model.config.use_per_frame_progress_token,
        max_frames=None,
    )

    all_ok = True
    for ex in EXAMPLES:
        video_path = examples_dir / ex["video"]
        upstream_progress_path = examples_dir / ex["progress_npy"]
        upstream_success_path = examples_dir / ex["success_npy"]

        missing = [p for p in (video_path, upstream_progress_path, upstream_success_path) if not p.exists()]
        if missing:
            print(f"[skip] {ex['name']}: missing {[str(m) for m in missing]}")
            all_ok = False
            continue

        print(f"\n=== {ex['name']} ===")
        print(f"  task: {ex['task']!r}")
        frames = _extract_frames_av(video_path, fps=args.fps)
        print(f"  decoded {frames.shape[0]} frames @ fps={args.fps}; shape={frames.shape}")

        progress, success = _run_lerobot(model, encoder, frames, ex["task"])

        upstream_progress = np.load(upstream_progress_path).astype(np.float32)
        upstream_success = np.load(upstream_success_path).astype(np.float32)

        progress_ok = _compare("progress", progress, upstream_progress, args.atol, args.rtol)
        success_ok = _compare("success", success, upstream_success, args.atol, args.rtol)
        verdict = "PASS" if (progress_ok and success_ok) else "FAIL"
        print(f"  -> {verdict}")
        all_ok = all_ok and progress_ok and success_ok

    print()
    if all_ok:
        print("All upstream example parity checks passed.")
        return 0
    print("Some upstream example parity checks FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
