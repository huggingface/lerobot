import logging
import threading
from pathlib import Path

import einops
import hydra
import imageio
import torch
from torchrl.data.replay_buffers import (
    SamplerWithoutReplacement,
)

from lerobot.common.datasets.factory import make_offline_buffer
from lerobot.common.logger import log_output_dir
from lerobot.common.utils import init_logging

NUM_EPISODES_TO_RENDER = 50
MAX_NUM_STEPS = 1000
FIRST_FRAME = 0


@hydra.main(version_base=None, config_name="default", config_path="../configs")
def visualize_dataset_cli(cfg: dict):
    visualize_dataset(cfg, out_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)


def cat_and_write_video(video_path, frames, fps):
    frames = torch.cat(frames)
    assert frames.dtype == torch.uint8
    frames = einops.rearrange(frames, "b c h w -> b h w c").numpy()
    imageio.mimsave(video_path, frames, fps=fps)


def visualize_dataset(cfg: dict, out_dir=None):
    if out_dir is None:
        raise NotImplementedError()

    init_logging()
    log_output_dir(out_dir)

    # we expect frames of each episode to be stored next to each others sequentially
    sampler = SamplerWithoutReplacement(
        shuffle=False,
    )

    logging.info("make_offline_buffer")
    offline_buffer = make_offline_buffer(
        cfg, overwrite_sampler=sampler, normalize=False, overwrite_batch_size=1, overwrite_prefetch=12
    )

    logging.info("Start rendering episodes from offline buffer")

    threads = []
    frames = {}
    current_ep_idx = 0
    logging.info(f"Visualizing episode {current_ep_idx}")
    for _ in range(MAX_NUM_STEPS * NUM_EPISODES_TO_RENDER):
        # TODO(rcadene): make it work with bsize > 1
        ep_td = offline_buffer.sample(1)
        ep_idx = ep_td["episode"][FIRST_FRAME].item()

        # TODO(rcaene): modify offline_buffer._sampler._sample_list or sampler to randomly sample an episode, but sequentially sample frames
        no_more_frames = offline_buffer._sampler._sample_list.numel() == 0
        new_episode = ep_idx != current_ep_idx

        if new_episode:
            logging.info(f"Visualizing episode {current_ep_idx}")

        for im_key in offline_buffer.image_keys:
            if new_episode or no_more_frames:
                # append last observed frames (the ones after last action taken)
                frames[im_key].append(ep_td[("next", *im_key)])

                video_dir = Path(out_dir) / "visualize_dataset"
                video_dir.mkdir(parents=True, exist_ok=True)

                if len(offline_buffer.image_keys) > 1:
                    camera = im_key[-1]
                    video_path = video_dir / f"episode_{current_ep_idx}_{camera}.mp4"
                else:
                    video_path = video_dir / f"episode_{current_ep_idx}.mp4"

                thread = threading.Thread(
                    target=cat_and_write_video,
                    args=(str(video_path), frames[im_key], cfg.fps),
                )
                thread.start()
                threads.append(thread)

                current_ep_idx = ep_idx

                # reset list of frames
                del frames[im_key]

            # append current cameras images to list of frames
            if im_key not in frames:
                frames[im_key] = []
            frames[im_key].append(ep_td[im_key])

        if no_more_frames:
            logging.info("Ran out of frames")
            break

    for thread in threads:
        thread.join()

    logging.info("End of visualize_dataset")


if __name__ == "__main__":
    visualize_dataset_cli()
