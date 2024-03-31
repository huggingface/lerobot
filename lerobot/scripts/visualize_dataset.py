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

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.logger import log_output_dir
from lerobot.common.utils import init_logging

NUM_EPISODES_TO_RENDER = 50
MAX_NUM_STEPS = 1000
FIRST_FRAME = 0


@hydra.main(version_base=None, config_name="default", config_path="../configs")
def visualize_dataset_cli(cfg: dict):
    visualize_dataset(cfg, out_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)


def cat_and_write_video(video_path, frames, fps):
    # Expects images in [0, 255].
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

    logging.info("make_dataset")
    dataset = make_dataset(
        cfg,
        overwrite_sampler=sampler,
        # remove all transformations such as rescale images from [0,255] to [0,1] or normalization
        normalize=False,
        overwrite_batch_size=1,
        overwrite_prefetch=12,
    )

    logging.info("Start rendering episodes from offline buffer")
    video_paths = render_dataset(dataset, out_dir, MAX_NUM_STEPS * NUM_EPISODES_TO_RENDER, cfg.fps)
    for video_path in video_paths:
        logging.info(video_path)


def render_dataset(dataset, out_dir, max_num_samples, fps):
    out_dir = Path(out_dir)
    video_paths = []
    threads = []
    frames = {}
    current_ep_idx = 0
    logging.info(f"Visualizing episode {current_ep_idx}")
    for i in range(max_num_samples):
        # TODO(rcadene): make it work with bsize > 1
        ep_td = dataset.sample(1)
        ep_idx = ep_td["episode"][FIRST_FRAME].item()

        # TODO(rcadene): modify dataset._sampler._sample_list or sampler to randomly sample an episode, but sequentially sample frames
        num_frames_left = dataset._sampler._sample_list.numel()
        episode_is_done = ep_idx != current_ep_idx

        if episode_is_done:
            logging.info(f"Rendering episode {current_ep_idx}")

        for im_key in dataset.image_keys:
            if not episode_is_done and num_frames_left > 0 and i < (max_num_samples - 1):
                # when first frame of episode, initialize frames dict
                if im_key not in frames:
                    frames[im_key] = []
                # add current frame to list of frames to render
                frames[im_key].append(ep_td[im_key])
            else:
                # When episode has no more frame in its list of observation,
                # one frame still remains. It is the result of the last action taken.
                # It is stored in `"next"`, so we add it to the list of frames to render.
                frames[im_key].append(ep_td["next"][im_key])

                out_dir.mkdir(parents=True, exist_ok=True)
                if len(dataset.image_keys) > 1:
                    camera = im_key[-1]
                    video_path = out_dir / f"episode_{current_ep_idx}_{camera}.mp4"
                else:
                    video_path = out_dir / f"episode_{current_ep_idx}.mp4"
                video_paths.append(str(video_path))

                thread = threading.Thread(
                    target=cat_and_write_video,
                    args=(str(video_path), frames[im_key], fps),
                )
                thread.start()
                threads.append(thread)

                current_ep_idx = ep_idx

                # reset list of frames
                del frames[im_key]

        if num_frames_left == 0:
            logging.info("Ran out of frames")
            break

        if current_ep_idx == NUM_EPISODES_TO_RENDER:
            break

    for thread in threads:
        thread.join()

    logging.info("End of visualize_dataset")
    return video_paths


if __name__ == "__main__":
    visualize_dataset_cli()
