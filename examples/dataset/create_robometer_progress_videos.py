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

"""Create videos with a Robometer progress overlay for one LeRobot dataset episode.

This is a lightweight smoke-test utility for Robometer checkpoints. It downloads
one episode video, samples a small number of frames, runs Robometer on those
frames, and reuses the progress overlay renderer from
``examples/dataset/create_progress_videos.py``.

Example:

    uv run python examples/dataset/create_robometer_progress_videos.py \\
        --repo-id lerobot/aloha_mobile_cabinet \\
        --episode 0 \\
        --reward-model-path lilkm/robometer-4b \\
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from examples.dataset.create_progress_videos import (
    composite_progress_video,
    convert_mp4_to_gif,
    download_episode_metadata,
    download_video_file,
    load_episode_meta,
)
from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel
from lerobot.rewards.robometer.modeling_robometer import decode_progress_outputs
from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep
from lerobot.utils.utils import init_logging


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def sample_episode_frames(
    video_path: Path,
    *,
    from_timestamp: float,
    to_timestamp: float,
    fps: float,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample RGB frames uniformly from an episode video segment.

    Returns:
        ``(frames, frame_indices)`` where ``frames`` is ``(T,H,W,C)`` uint8 RGB
        and ``frame_indices`` are local episode frame indices used for overlay.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    duration_seconds = to_timestamp - from_timestamp
    total_frames = max(int(round(duration_seconds * fps)), 1)
    frame_indices = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)

    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        for frame_idx in frame_indices:
            timestamp = from_timestamp + frame_idx / fps
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame_bgr = capture.read()
            if not ret:
                logging.warning("Could not read frame %d at %.3fs", frame_idx, timestamp)
                continue
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames could be sampled from {video_path}")

    return np.stack(frames), frame_indices[: len(frames)]


def predict_robometer_progress(
    frames: np.ndarray,
    *,
    task: str,
    reward_model_path: str,
    device: str,
) -> list[float]:
    """Run Robometer and return per-sampled-frame progress predictions."""
    config = RobometerConfig(pretrained_path=reward_model_path, device=device, max_frames=None)
    model = RobometerRewardModel.from_pretrained(reward_model_path, config=config)

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
    return decoded["progress_pred"][0]


def process_dataset(
    repo_id: str,
    episode: int,
    reward_model_path: str,
    device: str,
    camera_key: str | None,
    output_dir: Path,
    num_frames: int,
    task: str | None = None,
    create_gif: bool = False,
) -> Path:
    safe_name = repo_id.replace("/", "_")
    logging.info("Processing %s episode %d with Robometer %s", repo_id, episode, reward_model_path)

    local_path = download_episode_metadata(repo_id, episode)
    episode_meta = load_episode_meta(local_path, episode, camera_key)
    video_path = download_video_file(repo_id, local_path, episode_meta["video_rel"])

    task_name = task or episode_meta.get("task_name", "")
    if not task_name:
        raise ValueError("No task found in dataset metadata. Pass --task explicitly.")

    frames, frame_indices = sample_episode_frames(
        video_path,
        from_timestamp=episode_meta["from_ts"],
        to_timestamp=episode_meta["to_ts"],
        fps=episode_meta["fps"],
        num_frames=num_frames,
    )
    logging.info("Sampled %d frames for Robometer inference", len(frames))

    progress = predict_robometer_progress(
        frames,
        task=task_name,
        reward_model_path=reward_model_path,
        device=device,
    )
    progress_data = np.stack([frame_indices, np.asarray(progress, dtype=np.float32)], axis=1)
    logging.info("Progress predictions: %s", [round(float(value), 3) for value in progress])

    output_path = output_dir / f"{safe_name}_ep{episode}_robometer_progress.mp4"
    final_path = composite_progress_video(
        video_path=video_path,
        from_timestamp=episode_meta["from_ts"],
        to_timestamp=episode_meta["to_ts"],
        progress_data=progress_data,
        output_path=output_path,
        fps=episode_meta["fps"],
        task_name=task_name,
    )

    if create_gif:
        final_path = convert_mp4_to_gif(final_path)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create MP4/GIF videos with Robometer progress overlay for dataset episodes."
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face LeRobot dataset repo id.")
    parser.add_argument("--episode", type=int, required=True, help="Episode index to visualize.")
    parser.add_argument(
        "--reward-model-path",
        default="lilkm/robometer-4b",
        help="Robometer checkpoint path or Hub repo id (e.g. lilkm/robometer-4b).",
    )
    parser.add_argument("--device", default=_default_device(), help="Torch device for Robometer inference.")
    parser.add_argument(
        "--camera-key",
        default=None,
        help="Camera observation key (e.g. observation.images.top). Auto-selects first camera if omitted.",
    )
    parser.add_argument(
        "--task", default=None, help="Task description override if dataset metadata lacks one."
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of episode frames to sample for Robometer inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("progress_videos"),
        help="Directory to write output files.",
    )
    parser.add_argument("--gif", action="store_true", help="Also generate a GIF from the MP4 output.")
    args = parser.parse_args()

    init_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = process_dataset(
        repo_id=args.repo_id,
        episode=args.episode,
        reward_model_path=args.reward_model_path,
        device=args.device,
        camera_key=args.camera_key,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        task=args.task,
        create_gif=args.gif,
    )
    logging.info("Output: %s", result)


if __name__ == "__main__":
    main()
