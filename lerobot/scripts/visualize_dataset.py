import pickle
from pathlib import Path

import imageio
import simxarm
import torch
from torchrl.data.replay_buffers import (
    SamplerWithoutReplacement,
    SliceSampler,
    SliceSamplerWithoutReplacement,
)

from lerobot.common.datasets.simxarm import SimxarmExperienceReplay


def visualize_simxarm_dataset(dataset_id="xarm_lift_medium"):
    sampler = SliceSamplerWithoutReplacement(
        num_slices=1,
        strict_length=False,
        shuffle=False,
    )

    dataset = SimxarmExperienceReplay(
        dataset_id,
        # download="force",
        download=True,
        streaming=False,
        root="data",
        sampler=sampler,
    )

    NUM_EPISODES_TO_RENDER = 10
    MAX_NUM_STEPS = 50
    FIRST_FRAME = 0
    for _ in range(NUM_EPISODES_TO_RENDER):
        episode = dataset.sample(MAX_NUM_STEPS)

        ep_idx = episode["episode"][FIRST_FRAME].item()
        ep_frames = torch.cat(
            [
                episode["observation"]["image"][FIRST_FRAME][None, ...],
                episode["next", "observation"]["image"],
            ],
            dim=0,
        )

        video_dir = Path("tmp/2024_02_03_xarm_lift_medium")
        video_dir.mkdir(parents=True, exist_ok=True)
        # TODO(rcadene): make fps configurable
        video_path = video_dir / f"eval_episode_{ep_idx}.mp4"
        imageio.mimsave(video_path, ep_frames.numpy().transpose(0, 2, 3, 1), fps=15)

        # ran out of episodes
        if dataset._sampler._sample_list.numel() == 0:
            break


if __name__ == "__main__":
    visualize_simxarm_dataset()
