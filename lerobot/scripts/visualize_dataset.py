import logging
import threading
from pathlib import Path

import einops
import hydra
import imageio
import torch

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.logger import log_output_dir
from lerobot.common.utils.utils import init_logging

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

    logging.info("make_dataset")
    dataset = make_dataset(
        cfg,
        # remove all transformations such as rescale images from [0,255] to [0,1] or normalization
        normalize=False,
    )

    logging.info("Start rendering episodes from offline buffer")
    video_paths = render_dataset(dataset, out_dir, MAX_NUM_STEPS * NUM_EPISODES_TO_RENDER, cfg.fps)
    for video_path in video_paths:
        logging.info(video_path)


def render_dataset(dataset, out_dir, max_num_episodes):
    out_dir = Path(out_dir)
    video_paths = []
    threads = []

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=1,
        shuffle=False,
    )
    dl_iter = iter(dataloader)

    for ep_id in range(min(max_num_episodes, dataset.num_episodes)):
        logging.info(f"Rendering episode {ep_id}")

        frames = {}
        end_of_episode = False
        while not end_of_episode:
            item = next(dl_iter)

            for im_key in dataset.image_keys:
                # when first frame of episode, initialize frames dict
                if im_key not in frames:
                    frames[im_key] = []
                # add current frame to list of frames to render
                frames[im_key].append(item[im_key])

            end_of_episode = item["index"].item() == item["episode_data_index_to"].item() - 1

        out_dir.mkdir(parents=True, exist_ok=True)
        for im_key in dataset.image_keys:
            if len(dataset.image_keys) > 1:
                im_name = im_key.replace("observation.images.", "")
                video_path = out_dir / f"episode_{ep_id}_{im_name}.mp4"
            else:
                video_path = out_dir / f"episode_{ep_id}.mp4"
            video_paths.append(video_path)

            thread = threading.Thread(
                target=cat_and_write_video,
                args=(str(video_path), frames[im_key], dataset.fps),
            )
            thread.start()
            threads.append(thread)

    for thread in threads:
        thread.join()

    logging.info("End of visualize_dataset")
    return video_paths


if __name__ == "__main__":
    visualize_dataset_cli()
