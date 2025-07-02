#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib

import gymnasium as gym
import pytest

import lerobot
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
from tests.utils import require_env


@pytest.mark.parametrize("env_name, task_name", lerobot.env_task_pairs)
@require_env
def test_available_env_task(env_name: str, task_name: list):
    """
    This test verifies that all environments listed in `lerobot/__init__.py` can
    be successfully imported — if they're installed — and that their
    `available_tasks_per_env` are valid.
    """
    package_name = f"gym_{env_name}"
    importlib.import_module(package_name)
    gym_handle = f"{package_name}/{task_name}"
    assert gym_handle in gym.envs.registry, gym_handle


def test_available_policies():
    """
    This test verifies that the class attribute `name` for all policies is
    consistent with those listed in `lerobot/__init__.py`.
    """
    policy_classes = [ACTPolicy, DiffusionPolicy, TDMPCPolicy, VQBeTPolicy]
    policies = [pol_cls.name for pol_cls in policy_classes]
    assert set(policies) == set(lerobot.available_policies), policies


def test_print():
    print(lerobot.available_envs)
    print(lerobot.available_tasks_per_env)
    print(lerobot.available_datasets)
    print(lerobot.available_datasets_per_env)
    print(lerobot.available_real_world_datasets)
    print(lerobot.available_policies)
    print(lerobot.available_policies_per_env)
