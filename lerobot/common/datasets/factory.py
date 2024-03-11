import logging
import os
from pathlib import Path

import torch
from torchrl.data.replay_buffers import PrioritizedSliceSampler, SliceSampler

from lerobot.common.envs.transforms import NormalizeTransform

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))


def make_offline_buffer(
    cfg, overwrite_sampler=None, normalize=True, overwrite_batch_size=None, overwrite_prefetch=None
):
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
        from lerobot.common.datasets.simxarm import SimxarmExperienceReplay

        clsfunc = SimxarmExperienceReplay
        dataset_id = f"xarm_{cfg.env.task}_medium"

    elif cfg.env.name == "pusht":
        from lerobot.common.datasets.pusht import PushtExperienceReplay

        clsfunc = PushtExperienceReplay
        dataset_id = "pusht"

    elif cfg.env.name == "aloha":
        from lerobot.common.datasets.aloha import AlohaExperienceReplay

        clsfunc = AlohaExperienceReplay
        dataset_id = f"aloha_{cfg.env.task}"
    else:
        raise ValueError(cfg.env.name)

    offline_buffer = clsfunc(
        dataset_id=dataset_id,
        root=DATA_DIR,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=pin_memory,
        prefetch=prefetch if isinstance(prefetch, int) else None,
    )

    if normalize:
        # TODO(rcadene): make normalization strategy configurable between mean_std, min_max, manual_min_max, min_max_from_spec
        stats = offline_buffer.compute_or_load_stats()

        # we only normalize the state and action, since the images are usually normalized inside the model for now (except for tdmpc: see the following)
        in_keys = [("observation", "state"), ("action")]

        if cfg.policy.name == "tdmpc":
            for key in offline_buffer.image_keys:
                # TODO(rcadene): imagenet normalization is applied inside diffusion policy, but no normalization inside tdmpc
                in_keys.append(key)
                # since we use next observations in tdmpc
                in_keys.append(("next", *key))
            in_keys.append(("next", "observation", "state"))

        if cfg.policy.name == "diffusion" and cfg.env.name == "pusht":
            # TODO(rcadene): we overwrite stats to have the same as pretrained model, but we should remove this
            stats["observation", "state", "min"] = torch.tensor([13.456424, 32.938293], dtype=torch.float32)
            stats["observation", "state", "max"] = torch.tensor([496.14618, 510.9579], dtype=torch.float32)
            stats["action", "min"] = torch.tensor([12.0, 25.0], dtype=torch.float32)
            stats["action", "max"] = torch.tensor([511.0, 511.0], dtype=torch.float32)

        if (
            cfg.env.name == "aloha"
            and cfg.env.task == "sim_transfer_cube_scripted"
            and cfg.policy.name == "act"
        ):
            import pickle

            with open(
                "/home/rcadene/code/act/tmp/2024_03_10_sim_transfer_cube_scripted/dataset_stats.pkl", "rb"
            ) as file:
                dataset_stats = pickle.load(file)

            stats["action", "mean"] = torch.from_numpy(dataset_stats["action_mean"])[None, :]
            stats["action", "std"] = torch.from_numpy(dataset_stats["action_std"])[None, :]
            stats["observation", "state", "mean"] = torch.from_numpy(dataset_stats["qpos_mean"])[None, :]
            stats["observation", "state", "std"] = torch.from_numpy(dataset_stats["qpos_std"])[None, :]

        # TODO(rcadene): remove this and put it in config. Ideally we want to reproduce SOTA results just with mean_std
        normalization_mode = "mean_std" if cfg.env.name == "aloha" else "min_max"
        transform = NormalizeTransform(stats, in_keys, mode=normalization_mode)
        offline_buffer.set_transform(transform)

    if not overwrite_sampler:
        index = torch.arange(0, offline_buffer.num_samples, 1)
        sampler.extend(index)

    return offline_buffer
