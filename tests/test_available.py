"""
This test verifies that all environments, datasets, policies listed in `lerobot/__init__.py` can be sucessfully
imported and that their class attributes (eg. `available_datasets`, `name`, `available_tasks`) corresponds.

Note:
    When implementing a concrete class (e.g. `AlohaDataset`, `PushtEnv`, `DiffusionPolicy`), you need to:
        1. set the required class attributes:
            - for classes inheriting from `AbstractDataset`: `available_datasets`
            - for classes inheriting from `AbstractEnv`: `name`, `available_tasks`
            - for classes inheriting from `AbstractPolicy`: `name`
        2. update variables in `lerobot/__init__.py` (e.g. `available_envs`, `available_datasets_per_envs`, `available_policies`)
        3. update variables in `tests/test_available.py` by importing your new class
"""

import pytest
import lerobot

from lerobot.common.envs.aloha.env import AlohaEnv
from lerobot.common.envs.pusht.env import PushtEnv
from lerobot.common.envs.simxarm.env import SimxarmEnv

from lerobot.common.datasets.simxarm import SimxarmDataset
from lerobot.common.datasets.aloha import AlohaDataset
from lerobot.common.datasets.pusht import PushtDataset

from lerobot.common.policies.act.policy import ActionChunkingTransformerPolicy
from lerobot.common.policies.diffusion.policy import DiffusionPolicy
from lerobot.common.policies.tdmpc.policy import TDMPCPolicy


def test_available():
    pol_classes = [
        ActionChunkingTransformerPolicy,
        DiffusionPolicy,
        TDMPCPolicy,
    ]

    env_classes = [
        AlohaEnv,
        PushtEnv,
        SimxarmEnv,
    ]

    dat_classes = [
        AlohaDataset,
        PushtDataset,
        SimxarmDataset,
    ]
    
    policies = [pol_cls.name for pol_cls in pol_classes]
    assert set(policies) == set(lerobot.available_policies)

    envs = [env_cls.name for env_cls in env_classes]
    assert set(envs) == set(lerobot.available_envs)

    tasks_per_env = {env_cls.name: env_cls.available_tasks for env_cls in env_classes}
    for env in envs:
        assert set(tasks_per_env[env]) == set(lerobot.available_tasks_per_env[env])

    datasets_per_env = {env_cls.name: dat_cls.available_datasets for env_cls, dat_cls in zip(env_classes, dat_classes)}
    for env in envs:
        assert set(datasets_per_env[env]) == set(lerobot.available_datasets_per_env[env])

    
