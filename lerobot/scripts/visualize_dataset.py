import pickle
from pathlib import Path

import hydra
import imageio
import simxarm
import torch
from torchrl.data.replay_buffers import (
    SamplerWithoutReplacement,
    SliceSampler,
    SliceSamplerWithoutReplacement,
)

from lerobot.common.datasets.factory import make_offline_buffer


@hydra.main(version_base=None, config_name="default", config_path="../configs")
def visualize_dataset_cli(cfg: dict):
    visualize_dataset(
        cfg, out_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )


def visualize_dataset(cfg: dict, out_dir=None):
    if out_dir is None:
        raise NotImplementedError()

    sampler = SliceSamplerWithoutReplacement(
        num_slices=1,
        strict_length=False,
        shuffle=False,
    )

    offline_buffer = make_offline_buffer(cfg, sampler)

    NUM_EPISODES_TO_RENDER = 10
    MAX_NUM_STEPS = 1000
    FIRST_FRAME = 0
    for _ in range(NUM_EPISODES_TO_RENDER):
        episode = offline_buffer.sample(MAX_NUM_STEPS)

        ep_idx = episode["episode"][FIRST_FRAME].item()
        ep_frames = torch.cat(
            [
                episode["observation"]["image"][FIRST_FRAME][None, ...],
                episode["next", "observation"]["image"],
            ],
            dim=0,
        )

        video_dir = Path(out_dir) / "visualize_dataset"
        video_dir.mkdir(parents=True, exist_ok=True)
        # TODO(rcadene): make fps configurable
        video_path = video_dir / f"episode_{ep_idx}.mp4"

        assert ep_frames.min().item() >= 0
        assert ep_frames.max().item() > 1, "Not mendatory, but sanity check"
        assert ep_frames.max().item() <= 255
        ep_frames = ep_frames.type(torch.uint8)
        imageio.mimsave(
            video_path, ep_frames.numpy().transpose(0, 2, 3, 1), fps=cfg.fps
        )

        # ran out of episodes
        if offline_buffer._sampler._sample_list.numel() == 0:
            break


if __name__ == "__main__":
    visualize_dataset_cli()
