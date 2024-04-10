import logging
import os
from pathlib import Path

import torch
from torchvision.transforms import v2

from lerobot.common.datasets.utils import compute_stats
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
    if cfg.env.name == "xarm":
        from lerobot.common.datasets.xarm import XarmDataset

        clsfunc = XarmDataset

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
        # TODO(rcadene): remove this and put it in config. Ideally we want to reproduce SOTA results just with mean_std
        normalization_mode = "mean_std" if cfg.env.name == "aloha" else "min_max"

        if cfg.policy.name == "diffusion" and cfg.env.name == "pusht":
            stats = {}
            # TODO(rcadene): we overwrite stats to have the same as pretrained model, but we should remove this
            stats["observation.state"] = {}
            stats["observation.state"]["min"] = torch.tensor([13.456424, 32.938293], dtype=torch.float32)
            stats["observation.state"]["max"] = torch.tensor([496.14618, 510.9579], dtype=torch.float32)
            stats["action"] = {}
            stats["action"]["min"] = torch.tensor([12.0, 25.0], dtype=torch.float32)
            stats["action"]["max"] = torch.tensor([511.0, 511.0], dtype=torch.float32)
        elif stats_path is None:
            # instantiate a one frame dataset with light transform
            stats_dataset = clsfunc(
                dataset_id=cfg.dataset_id,
                root=DATA_DIR,
                transform=Prod(in_keys=clsfunc.image_keys, prod=1 / 255.0),
            )

            # load stats if the file exists already or compute stats and save it
            precomputed_stats_path = stats_dataset.data_dir / "stats.pth"
            if precomputed_stats_path.exists():
                stats = torch.load(precomputed_stats_path)
            else:
                logging.info(f"compute_stats and save to {precomputed_stats_path}")
                stats = compute_stats(stats_dataset)
                torch.save(stats, stats_path)
        else:
            stats = torch.load(stats_path)

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

    delta_timestamps = cfg.policy.get("delta_timestamps")
    if delta_timestamps is not None:
        for key in delta_timestamps:
            if isinstance(delta_timestamps[key], str):
                delta_timestamps[key] = eval(delta_timestamps[key])

    dataset = clsfunc(
        dataset_id=cfg.dataset_id,
        root=DATA_DIR,
        delta_timestamps=delta_timestamps,
        transform=transforms,
    )

    return dataset
