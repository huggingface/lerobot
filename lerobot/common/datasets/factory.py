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

        # # TODO(now): These stats are needed to use their pretrained model for sim_transfer_cube_human.
        # # (Pdb) stats['observation']['state']['mean']
        # # tensor([-0.0071, -0.6293,  1.0351, -0.0517, -0.4642, -0.0754,  0.4751, -0.0373,
        # #         -0.3324,  0.9034, -0.2258, -0.3127, -0.2412,  0.6866])
        # stats["observation", "state", "mean"] = torch.tensor(
        #     [
        #         -0.00740268,
        #         -0.63187766,
        #         1.0356655,
        #         -0.05027218,
        #         -0.46199223,
        #         -0.07467502,
        #         0.47467607,
        #         -0.03615446,
        #         -0.33203387,
        #         0.9038929,
        #         -0.22060776,
        #         -0.31011587,
        #         -0.23484458,
        #         0.6842416,
        #     ]
        # )
        # # (Pdb) stats['observation']['state']['std']
        # # tensor([0.0022, 0.0520, 0.0291, 0.0092, 0.0267, 0.0145, 0.0563, 0.0179, 0.0494,
        # #         0.0326, 0.0476, 0.0535, 0.0956, 0.0513])
        # stats["observation", "state", "std"] = torch.tensor(
        #     [
        #         0.01219023,
        #         0.2975381,
        #         0.16728032,
        #         0.04733803,
        #         0.1486037,
        #         0.08788499,
        #         0.31752336,
        #         0.1049916,
        #         0.27933604,
        #         0.18094037,
        #         0.26604933,
        #         0.30466506,
        #         0.5298686,
        #         0.25505227,
        #     ]
        # )
        # # (Pdb) stats['action']['mean']
        # # tensor([-0.0075, -0.6346,  1.0353, -0.0465, -0.4686, -0.0738,  0.3723, -0.0396,
        # #         -0.3184,  0.8991, -0.2065, -0.3182, -0.2338,  0.5593])
        # stats["action"]["mean"] = torch.tensor(
        #     [
        #         -0.00756444,
        #         -0.6281845,
        #         1.0312834,
        #         -0.04664314,
        #         -0.47211358,
        #         -0.074527,
        #         0.37389806,
        #         -0.03718753,
        #         -0.3261143,
        #         0.8997205,
        #         -0.21371077,
        #         -0.31840396,
        #         -0.23360962,
        #         0.551947,
        #     ]
        # )
        # # (Pdb) stats['action']['std']
        # # tensor([0.0023, 0.0514, 0.0290, 0.0086, 0.0263, 0.0143, 0.0593, 0.0185, 0.0510,
        # #         0.0328, 0.0478, 0.0531, 0.0945, 0.0794])
        # stats["action"]["std"] = torch.tensor(
        #     [
        #         0.01252818,
        #         0.2957442,
        #         0.16701928,
        #         0.04584508,
        #         0.14833844,
        #         0.08763024,
        #         0.30665937,
        #         0.10600077,
        #         0.27572668,
        #         0.1805853,
        #         0.26304692,
        #         0.30708534,
        #         0.5305411,
        #         0.38381037,
        #     ]
        # )
        # transforms.append(NormalizeTransform(stats, in_keys, mode=normalization_mode))  # noqa: F821

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
        delta_timestamps = {
            "observation.images.top": [0],
            "observation.state": [0],
            "action": [i / clsfunc.fps for i in range(cfg.policy.horizon)],
        }

    dataset = clsfunc(
        dataset_id=cfg.dataset_id,
        root=DATA_DIR,
        delta_timestamps=delta_timestamps,
        transform=transforms,
    )

    return dataset
