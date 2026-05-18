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
        --decoder decord

The number of frames sampled per video is derived from the length of each
upstream ``.npy`` reference, so the script does not need a ``--fps`` argument
(the README documents ``fps=3`` for SOAR / Berkeley, but the Jaco Play
reference was generated with a different fps).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel
from lerobot.rewards.robometer.modeling_robometer import decode_progress_outputs
from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep

try:
    import decord  # type: ignore

    _HAS_DECORD = True
except ImportError:
    decord = None  # type: ignore
    _HAS_DECORD = False

try:
    import av

    _HAS_AV = True
except ImportError:
    av = None  # type: ignore
    _HAS_AV = False

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


def _extract_frames_decord(video_path: Path, num_frames: int) -> tuple[np.ndarray, str]:
    """Sample ``num_frames`` indices uniformly from the video using decord.

    Mirrors upstream's ``extract_frames`` indexing
    (``third_party/robometer/scripts/example_inference.py``): a
    ``np.linspace(0, total_frames-1, num_frames)`` lookup over decord's
    ``VideoReader``. We pass ``num_frames`` explicitly (derived from the
    upstream reference output length) so we don't have to guess what ``fps``
    upstream actually used when generating each saved ``.npy`` — the file
    length is the ground truth.
    """
    vr = decord.VideoReader(str(video_path), num_threads=1)
    total_frames = len(vr)
    if total_frames == 0:
        raise RuntimeError(f"No decodable frames in {video_path}.")
    desired_frames = max(1, min(int(num_frames), total_frames))
    indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()
    native_fps = float(vr.get_avg_fps()) or 1.0
    return frames, f"decord total={total_frames} native_fps={native_fps:.3f}"


def _extract_frames_av(video_path: Path, num_frames: int) -> tuple[np.ndarray, str]:
    """PyAV fallback for environments without decord.

    PyAV and decord can disagree on ``total_frames`` for the same container,
    so the sampled frame indices can drift. Install ``decord`` for a real
    parity check; this fallback is for smoke tests only.
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
    desired_frames = max(1, min(int(num_frames), total_frames))
    indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int)
    frames = np.stack([rgb_frames[i] for i in indices])
    return frames, f"av total={total_frames} native_fps={native_fps:.3f}"


def _extract_frames(video_path: Path, num_frames: int, prefer: str) -> tuple[np.ndarray, str]:
    """Decoder dispatch. ``prefer`` is ``"decord"`` | ``"av"`` | ``"auto"``."""
    if prefer == "decord":
        if not _HAS_DECORD:
            raise RuntimeError("decord requested but not installed (`uv pip install decord`).")
        return _extract_frames_decord(video_path, num_frames)
    if prefer == "av":
        if not _HAS_AV:
            raise RuntimeError("av requested but not installed.")
        return _extract_frames_av(video_path, num_frames)
    # auto
    if _HAS_DECORD:
        return _extract_frames_decord(video_path, num_frames)
    if _HAS_AV:
        return _extract_frames_av(video_path, num_frames)
    raise RuntimeError("No video decoder available (install `decord` or `av`).")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; returns 1.0 for constant inputs (no signal to align)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.size < 2:
        return 1.0
    da = a - a.mean()
    db = b - b.mean()
    denom = float(np.sqrt((da * da).sum()) * np.sqrt((db * db).sum()))
    if denom == 0:
        return 1.0
    return float((da * db).sum() / denom)


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


def _compare(
    name: str,
    lerobot: np.ndarray,
    upstream: np.ndarray,
    *,
    atol: float,
    pearson_min: float,
) -> bool:
    if lerobot.shape != upstream.shape:
        print(f"  {name:8s}  SHAPE MISMATCH lerobot={lerobot.shape} upstream={upstream.shape}")
        return False
    abs_diff = np.abs(lerobot - upstream)
    pearson = _pearson(lerobot, upstream)
    abs_ok = bool(abs_diff.max() <= atol)
    pearson_ok = bool(pearson >= pearson_min)
    verdict = "PASS" if (abs_ok or pearson_ok) else "FAIL"
    print(
        f"  {name:8s}  shape={lerobot.shape}  max|Δ|={abs_diff.max():.3e}  "
        f"mean|Δ|={abs_diff.mean():.3e}  pearson={pearson:.4f}  "
        f"(atol={atol:.0e} pearson_min={pearson_min:.3f}) -> {verdict}"
    )
    return abs_ok or pearson_ok


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
        "--decoder",
        choices=("auto", "decord", "av"),
        default="auto",
        help=(
            "Video decoder. ``auto`` prefers decord (matches upstream) and falls back to av. "
            "Force ``decord`` for a clean parity check."
        ),
    )
    parser.add_argument(
        "--progress-atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for the progress array. Default 1e-2 covers CUDA bf16 noise.",
    )
    parser.add_argument(
        "--success-atol",
        type=float,
        default=1e-1,
        help=(
            "Absolute tolerance for the success array. Looser than progress because "
            "``sigmoid`` amplifies logit-space noise near 0.5."
        ),
    )
    parser.add_argument(
        "--pearson-min",
        type=float,
        default=0.99,
        help="Minimum Pearson correlation for a PASS verdict (per array).",
    )
    args = parser.parse_args()

    if args.decoder == "av" or (args.decoder == "auto" and not _HAS_DECORD):
        print(
            "WARNING: using PyAV decoder. PyAV's total-frame count can differ from decord's, "
            "which propagates into different sampled-frame indices. Install `decord` and "
            "re-run for a clean parity check.",
            file=sys.stderr,
        )

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

        # Trust the upstream reference array as the source of truth for how
        # many frames to sample. The README documents fps=3 for SOAR/Berkeley
        # but Jaco Play was generated with a different fps, so any hardcoded
        # ``--fps`` mismatches at least one example. The npy length always
        # tells us what upstream actually used.
        upstream_progress = np.load(upstream_progress_path).astype(np.float32)
        upstream_success = np.load(upstream_success_path).astype(np.float32)
        target_num_frames = int(upstream_progress.shape[0])
        frames, decoder_info = _extract_frames(video_path, target_num_frames, prefer=args.decoder)
        print(
            f"  decoded {frames.shape[0]} frames (matches upstream npy length); "
            f"shape={frames.shape}  [{decoder_info}]"
        )

        progress, success = _run_lerobot(model, encoder, frames, ex["task"])

        progress_ok = _compare(
            "progress",
            progress,
            upstream_progress,
            atol=args.progress_atol,
            pearson_min=args.pearson_min,
        )
        success_ok = _compare(
            "success",
            success,
            upstream_success,
            atol=args.success_atol,
            pearson_min=args.pearson_min,
        )
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
