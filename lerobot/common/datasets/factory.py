import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

DATA_DIR = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else None


def make_dataset(
    cfg,
    split="train",
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

    delta_timestamps = cfg.policy.get("delta_timestamps")
    if delta_timestamps is not None:
        for key in delta_timestamps:
            if isinstance(delta_timestamps[key], str):
                delta_timestamps[key] = eval(delta_timestamps[key])

    # TODO(rcadene): add data augmentations

    dataset = clsfunc(
        dataset_id=cfg.dataset_id,
        split=split,
        root=DATA_DIR,
        delta_timestamps=delta_timestamps,
    )

    if cfg.get("override_dataset_stats"):
        for key, stats_dict in cfg.override_dataset_stats.items():
            for stats_type, listconfig in stats_dict.items():
                # example of stats_type: min, max, mean, std
                stats = OmegaConf.to_container(listconfig, resolve=True)
                dataset.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
