import os
from pathlib import Path

import torch
from torchvision.transforms import v2

from lerobot.common.datasets.utils import compute_or_load_stats
from lerobot.common.transforms import NormalizeTransform, Prod

# DATA_DIR specifies to location where datasets are loaded. By default, DATA_DIR is None and
# we load from `$HOME/.cache/huggingface/hub/datasets`. For our unit tests, we set `DATA_DIR=tests/data`
# to load a subset of our datasets for faster continuous integration.
DATA_DIR = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else None


def make_dataset(
    cfg,
    # set normalize=False to remove all transformations and keep images unnormalized in [0,255]
    normalize=True,
    stats_path=None,
):
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

    transforms = None
    if normalize:
        # TODO(rcadene): make normalization strategy configurable between mean_std, min_max, manual_min_max,
        # min_max_from_spec
        # stats = dataset.compute_or_load_stats() if stats_path is None else torch.load(stats_path)

        if cfg.policy.name == "diffusion" and cfg.env.name == "pusht":
            stats = {}
            # TODO(rcadene): we overwrite stats to have the same as pretrained model, but we should remove this
            stats["observation.state"] = {}
            stats["observation.state"]["min"] = torch.tensor([13.456424, 32.938293], dtype=torch.float32)
            stats["observation.state"]["max"] = torch.tensor([496.14618, 510.9579], dtype=torch.float32)
            stats["action"] = {}
            stats["action"]["min"] = torch.tensor([12.0, 25.0], dtype=torch.float32)
            stats["action"]["max"] = torch.tensor([511.0, 511.0], dtype=torch.float32)
        else:
            # instantiate a one frame dataset with light transform
            stats_dataset = clsfunc(
                dataset_id=cfg.dataset_id,
                root=DATA_DIR,
                transform=Prod(in_keys=clsfunc.image_keys, prod=1 / 255.0),
            )
            stats = compute_or_load_stats(stats_dataset)

        # TODO(rcadene): remove this and put it in config. Ideally we want to reproduce SOTA results just with mean_std
        normalization_mode = "mean_std" if cfg.env.name == "aloha" else "min_max"

        transforms = v2.Compose(
            [
                # TODO(rcadene): we need to do something about image_keys
                Prod(in_keys=clsfunc.image_keys, prod=1 / 255.0),
                NormalizeTransform(
                    stats,
                    in_keys=[
                        "observation.state",
                        "action",
                    ],
                    mode=normalization_mode,
                ),
            ]
        )

    if cfg.policy.name == "diffusion" and cfg.env.name == "pusht":
        # TODO(rcadene): implement delta_timestamps in config
        delta_timestamps = {
            "observation.image": [-0.1, 0],
            "observation.state": [-0.1, 0],
            "action": [-0.1] + [i / clsfunc.fps for i in range(15)],
        }
    else:
        delta_timestamps = None

    dataset = clsfunc(
        dataset_id=cfg.dataset_id,
        root=DATA_DIR,
        delta_timestamps=delta_timestamps,
        transform=transforms,
    )

    return dataset
