"""
This test verifies that all environments, datasets, policies listed in `lerobot/__init__.py` can be sucessfully
imported and that their class attributes (eg. `available_datasets`, `name`, `available_tasks`) are valid.

When implementing a new dataset (e.g. `AlohaDataset`), policy (e.g. `DiffusionPolicy`), or environment, follow these steps:
- Set the required class attributes: `available_datasets`.
- Set the required class attributes: `name`.
- Update variables in `lerobot/__init__.py` (e.g. `available_envs`, `available_datasets_per_envs`, `available_policies`)
- Update variables in `tests/test_available.py` by importing your new class
"""

import importlib
import pytest
import lerobot
import gymnasium as gym

from lerobot.common.datasets.xarm import XarmDataset
from lerobot.common.datasets.aloha import AlohaDataset
from lerobot.common.datasets.pusht import PushtDataset

from lerobot.common.policies.act.modeling_act import ActionChunkingTransformerPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.tdmpc.policy import TDMPCPolicy


def test_available():
    policy_classes = [
        ActionChunkingTransformerPolicy,
        DiffusionPolicy,
        TDMPCPolicy,
    ]

    dataset_class_per_env = {
        "aloha": AlohaDataset,
        "pusht": PushtDataset,
        "xarm": XarmDataset,
    }
    
    policies = [pol_cls.name for pol_cls in policy_classes]
    assert set(policies) == set(lerobot.available_policies), policies

    for env_name in lerobot.available_envs:
        for task_name in lerobot.available_tasks_per_env[env_name]:
            package_name = f"gym_{env_name}"
            importlib.import_module(package_name)
            gym_handle = f"{package_name}/{task_name}"
            assert gym_handle in gym.envs.registry.keys(), gym_handle

        dataset_class = dataset_class_per_env[env_name]
        available_datasets = lerobot.available_datasets_per_env[env_name]
        assert set(available_datasets) == set(dataset_class.available_datasets), f"{env_name=} {available_datasets=}"

    
