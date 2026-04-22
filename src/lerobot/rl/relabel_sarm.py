#!/usr/bin/env python
"""Relabel an offline dataset's reward channel with SARM.

Produces a new LeRobotDataset where every transition's reward is recomputed by
running a trained SARM model over that frame's observation window. Use this
to make the offline HIL-SERL replay buffer consistent with the online reward
signal that ``SARMRewardProcessorStep`` emits.

Reuses ``SARMRewardProcessorStep`` directly, so window construction (ring
buffer past + replicated current for future slots) is byte-for-byte identical
to what the actor sees at runtime.

``--reward-mode`` MUST match the ``reward_mode`` used in the online RL config.
``next.done`` is preserved from the source (progress-threshold crossings
mid-episode do not force done=True).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.reward_model.sarm import (
    SARMRewardConfig,
    SARMRewardProcessorStep,
)
from lerobot.utils.constants import DONE, REWARD

logger = logging.getLogger(__name__)

_SKIP_KEYS = {"task_index", "timestamp", "episode_index", "frame_index", "index", "task"}


def _build_new_frame(frame: dict, new_reward: float, new_done: bool) -> dict:
    new_frame: dict = {}
    for key, value in frame.items():
        if key in _SKIP_KEYS:
            continue
        if key == REWARD:
            value = torch.tensor([float(new_reward)], dtype=torch.float32)
        elif key == DONE:
            value = torch.tensor([bool(new_done)], dtype=torch.bool)
        elif key.startswith("complementary_info") and hasattr(value, "dim") and value.dim() == 0:
            value = value.unsqueeze(0)
        new_frame[key] = value
    return new_frame


def _episode_boundaries(dataset: LeRobotDataset) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    for ep in dataset.meta.episodes:
        bounds.append((int(ep["dataset_from_index"]), int(ep["dataset_to_index"])))
    return bounds


def relabel(
    src_repo_id: str,
    sarm_checkpoint: str,
    reward_mode: str = "delta",
    head_mode: str = "sparse",
    src_root: str | None = None,
    new_repo_id: str | None = None,
    task_text: str | None = None,
    success_threshold: float = 0.9,
    device: str = "cuda",
) -> LeRobotDataset:
    src = LeRobotDataset(repo_id=src_repo_id, root=src_root)

    if task_text is None:
        task_text = src[0].get("task")
        if task_text is None:
            raise ValueError(
                "src[0] has no 'task' key. Pass --task '' (empty) or the exact string SARM was trained with."
            )
        if not isinstance(task_text, str):
            raise ValueError(
                f"src[0]['task'] is not a string (got {type(task_text).__name__}). Pass --task explicitly."
            )
    logger.info("SARM task text (fed to CLIP): %r", task_text)

    if new_repo_id is None:
        new_repo_id = f"{src_repo_id}_sarm_{reward_mode}"
    new_root = Path(str(src.root) + f"_sarm_{reward_mode}")

    # stats_dataset_repo_id pinned to source — source has populated stats by
    # construction; a split dataset may have empty stats, producing OOD input.
    sarm_step = SARMRewardProcessorStep(
        config=SARMRewardConfig(
            pretrained_path=sarm_checkpoint,
            device=device,
            task=task_text,
            head_mode=head_mode,
            reward_mode=reward_mode,
            success_threshold=success_threshold,
            stats_dataset_repo_id=src_repo_id,
        ),
        terminate_on_success=False,
    )

    common_kwargs = {
        "fps": int(src.fps),
        "robot_type": src.meta.robot_type,
        "features": src.meta.info["features"],
        "use_videos": len(src.meta.video_keys) > 0,
    }
    new_ds = LeRobotDataset.create(repo_id=new_repo_id, root=new_root, **common_kwargs)

    bounds = _episode_boundaries(src)
    n_frames = 0
    reward_sum = 0.0
    n_threshold_frames = 0
    progress_at_terminal: list[float] = []

    for _ep_idx, (ep_start, ep_end) in enumerate(tqdm(bounds, desc=f"relabeling ({reward_mode})")):
        sarm_step.reset()

        for frame_idx in range(ep_start, ep_end):
            frame = src[frame_idx]
            is_terminal = frame_idx == ep_end - 1

            observation = {
                k: v
                for k, v in frame.items()
                if isinstance(v, torch.Tensor) and k.startswith("observation.")
            }

            # Use SYNCHRONOUS compute_reward (avoids async ring-buffer race in batched relabel).
            progress = sarm_step.compute_reward(observation)

            if reward_mode == "dense":
                reward = progress
            elif reward_mode == "delta":
                reward = progress - sarm_step._prev_progress
            else:  # binary
                reward = 1.0 if progress >= success_threshold else 0.0
            sarm_step._prev_progress = progress

            done = is_terminal
            new_frame = _build_new_frame(frame, new_reward=reward, new_done=done)
            new_frame["task"] = frame.get("task", task_text)
            new_ds.add_frame(new_frame)

            n_frames += 1
            reward_sum += reward
            if progress >= success_threshold:
                n_threshold_frames += 1
            if is_terminal:
                progress_at_terminal.append(progress)

        new_ds.save_episode()

    new_ds.finalize()

    meta_dir = new_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "sarm_relabel.json").write_text(
        json.dumps(
            {
                "source_repo_id": src_repo_id,
                "sarm_checkpoint": str(sarm_checkpoint),
                "reward_mode": reward_mode,
                "success_threshold": success_threshold,
                "task_text": task_text,
            },
            indent=2,
        )
    )

    mean_terminal_progress = (
        sum(progress_at_terminal) / len(progress_at_terminal) if progress_at_terminal else 0.0
    )
    print(
        f"\nSARM relabel done:\n"
        f"  source: {src_repo_id} ({len(src)} frames, {len(bounds)} episodes)\n"
        f"  dest:   {new_repo_id}  ({new_root})\n"
        f"  mode:   {reward_mode}  task={task_text!r}\n"
        f"  frames written:      {n_frames}\n"
        f"  total reward:        {reward_sum:.3f}\n"
        f"  mean reward / frame: {reward_sum / max(1, n_frames):.4f}\n"
        f"  frames ≥ threshold:  {n_threshold_frames}/{n_frames}  (threshold={success_threshold})\n"
        f"  mean terminal progress: {mean_terminal_progress:.3f}\n"
    )
    return new_ds


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Relabel a LeRobotDataset with SARM-derived rewards.")
    parser.add_argument("--src-repo-id", type=str, required=True)
    parser.add_argument("--src-root", type=str, default=None)
    parser.add_argument("--sarm-checkpoint", type=str, required=True)
    parser.add_argument("--reward-mode", type=str, default="delta", choices=["binary", "dense", "delta"])
    parser.add_argument("--new-repo-id", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--head-mode", type=str, default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--success-threshold", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    relabel(
        src_repo_id=args.src_repo_id,
        src_root=args.src_root,
        sarm_checkpoint=args.sarm_checkpoint,
        reward_mode=args.reward_mode,
        head_mode=args.head_mode,
        new_repo_id=args.new_repo_id,
        task_text=args.task,
        success_threshold=args.success_threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
