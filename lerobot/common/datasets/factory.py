import logging

import torch
from omegaconf import OmegaConf

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def make_dataset(
    cfg,
    split="train",
):
    if cfg.env.name not in cfg.dataset_repo_id:
        logging.warning(
            f"There might be a mismatch between your training dataset ({cfg.dataset_repo_id=}) and your "
            f"environment ({cfg.env.name=})."
        )

    # TODO(rcadene): add data augmentations

    dataset = LeRobotDataset(cfg.dataset_repo_id, split=split)

    if cfg.get("override_dataset_stats"):
        for key, stats_dict in cfg.override_dataset_stats.items():
            for stats_type, listconfig in stats_dict.items():
                # example of stats_type: min, max, mean, std
                stats = OmegaConf.to_container(listconfig, resolve=True)
                dataset.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
