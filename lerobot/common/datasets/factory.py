import logging
import os
from pathlib import Path

import torch
from torchrl.data.replay_buffers import PrioritizedSliceSampler, SliceSampler

from lerobot.common.transforms import NormalizeTransform, Prod

# DATA_DIR specifies to location where datasets are loaded. By default, DATA_DIR is None and
# we load from `$HOME/.cache/huggingface/hub/datasets`. For our unit tests, we set `DATA_DIR=tests/data`
# to load a subset of our datasets for faster continuous integration.
DATA_DIR = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else None


def make_offline_buffer(
    cfg,
    overwrite_sampler=None,
    # set normalize=False to remove all transformations and keep images unnormalized in [0,255]
    normalize=True,
    overwrite_batch_size=None,
    overwrite_prefetch=None,
    stats_path=None,
    # Don't actually load any data. This is a stand-in solution to get the transforms.
    dummy=False,
):
    if dummy and normalize and stats_path is None:
        raise ValueError("`stats_path` is required if `dummy` and `normalize` are True.")

    if cfg.policy.balanced_sampling:
        assert cfg.online_steps > 0
        batch_size = None
        pin_memory = False
        prefetch = None
    else:
        assert cfg.online_steps == 0
        num_slices = cfg.policy.batch_size
        batch_size = cfg.policy.horizon * num_slices
        pin_memory = cfg.device == "cuda"
        prefetch = cfg.prefetch

    if overwrite_batch_size is not None:
        batch_size = overwrite_batch_size

    if overwrite_prefetch is not None:
        prefetch = overwrite_prefetch

    if overwrite_sampler is None:
        # TODO(rcadene): move batch_size outside
        num_traj_per_batch = cfg.policy.batch_size  # // cfg.horizon
        # TODO(rcadene): Sampler outputs a batch_size <= cfg.batch_size.
        # We would need to add a transform to pad the tensordict to ensure batch_size == cfg.batch_size.

        if cfg.offline_prioritized_sampler:
            logging.info("use prioritized sampler for offline dataset")
            sampler = PrioritizedSliceSampler(
                max_capacity=100_000,
                alpha=cfg.policy.per_alpha,
                beta=cfg.policy.per_beta,
                num_slices=num_traj_per_batch,
                strict_length=False,
            )
        else:
            logging.info("use simple sampler for offline dataset")
            sampler = SliceSampler(
                num_slices=num_traj_per_batch,
                strict_length=False,
            )
    else:
        sampler = overwrite_sampler

    if cfg.env.name == "simxarm":
        from lerobot.common.datasets.simxarm import SimxarmDataset

        clsfunc = SimxarmDataset

    elif cfg.env.name == "pusht":
        from lerobot.common.datasets.pusht import PushtDataset

        clsfunc = PushtDataset

    elif cfg.env.name == "aloha":
        from lerobot.common.datasets.aloha import AlohaDataset

        clsfunc = AlohaDataset
    else:
        raise ValueError(cfg.env.name)

    # TODO(rcadene): backward compatiblity to load pretrained pusht policy
    dataset_id = cfg.get("dataset_id")
    if dataset_id is None and cfg.env.name == "pusht":
        dataset_id = "pusht"

    offline_buffer = clsfunc(
        dataset_id=dataset_id,
        sampler=sampler,
        batch_size=batch_size,
        root=DATA_DIR,
        pin_memory=pin_memory,
        prefetch=prefetch if isinstance(prefetch, int) else None,
        dummy=dummy,
    )

    if cfg.policy.name == "tdmpc":
        img_keys = []
        for key in offline_buffer.image_keys:
            img_keys.append(("next", *key))
        img_keys += offline_buffer.image_keys
    else:
        img_keys = offline_buffer.image_keys

    if normalize:
        transforms = [Prod(in_keys=img_keys, prod=1 / 255)]

        # TODO(rcadene): make normalization strategy configurable between mean_std, min_max, manual_min_max,
        # min_max_from_spec
        stats = offline_buffer.compute_or_load_stats() if stats_path is None else torch.load(stats_path)

        # we only normalize the state and action, since the images are usually normalized inside the model for
        # now (except for tdmpc: see the following)
        in_keys = [("observation", "state"), ("action")]

        if cfg.policy.name == "tdmpc":
            # TODO(rcadene): we add img_keys to the keys to normalize for tdmpc only, since diffusion and act policies normalize the image inside the model for now
            in_keys += img_keys
            # TODO(racdene): since we use next observations in tdmpc, we also add them to the normalization. We are wasting a bit of compute on this for now.
            in_keys += [("next", *key) for key in img_keys]
            in_keys.append(("next", "observation", "state"))

        if cfg.policy.name == "diffusion" and cfg.env.name == "pusht":
            # TODO(rcadene): we overwrite stats to have the same as pretrained model, but we should remove this
            stats["observation", "state", "min"] = torch.tensor([13.456424, 32.938293], dtype=torch.float32)
            stats["observation", "state", "max"] = torch.tensor([496.14618, 510.9579], dtype=torch.float32)
            stats["action", "min"] = torch.tensor([12.0, 25.0], dtype=torch.float32)
            stats["action", "max"] = torch.tensor([511.0, 511.0], dtype=torch.float32)

        # TODO(rcadene): remove this and put it in config. Ideally we want to reproduce SOTA results just with mean_std
        normalization_mode = "mean_std" if cfg.env.name == "aloha" else "min_max"
        transforms.append(NormalizeTransform(stats, in_keys, mode=normalization_mode))

        offline_buffer.set_transform(transforms)

    if not overwrite_sampler:
        index = torch.arange(0, offline_buffer.num_samples, 1)
        sampler.extend(index)

    return offline_buffer
