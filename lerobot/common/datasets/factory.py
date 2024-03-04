import logging
import os
from pathlib import Path

import torch
from torchrl.data.replay_buffers import PrioritizedSliceSampler, SliceSampler

from lerobot.common.datasets.pusht import PushtExperienceReplay
from lerobot.common.datasets.simxarm import SimxarmExperienceReplay

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

# TODO(rcadene): implement

# dataset_d4rl = D4RLExperienceReplay(
#     dataset_id="maze2d-umaze-v1",
#     split_trajs=False,
#     batch_size=1,
#     sampler=SamplerWithoutReplacement(drop_last=False),
#     prefetch=4,
#     direct_download=True,
# )

# dataset_openx = OpenXExperienceReplay(
#     "cmu_stretch",
#     batch_size=1,
#     num_slices=1,
#     #download="force",
#     streaming=False,
#     root="data",
# )


def make_offline_buffer(cfg, sampler=None):
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

    overwrite_sampler = sampler is not None

    if not overwrite_sampler:
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

    if cfg.env.name == "simxarm":
        # TODO(rcadene): add PrioritizedSliceSampler inside Simxarm to not have to `sampler.extend(index)` here
        offline_buffer = SimxarmExperienceReplay(
            f"xarm_{cfg.env.task}_medium",
            # download="force",
            download=True,
            streaming=False,
            root=str(DATA_DIR),
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=pin_memory,
            prefetch=prefetch if isinstance(prefetch, int) else None,
        )
    elif cfg.env.name == "pusht":
        offline_buffer = PushtExperienceReplay(
            "pusht",
            streaming=False,
            root=DATA_DIR,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=pin_memory,
            prefetch=prefetch if isinstance(prefetch, int) else None,
        )
    else:
        raise ValueError(cfg.env.name)

    if not overwrite_sampler:
        num_steps = len(offline_buffer)
        index = torch.arange(0, num_steps, 1)
        sampler.extend(index)

    return offline_buffer
