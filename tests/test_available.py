import importlib
import pytest
import lerobot
import gymnasium as gym

from lerobot.common.utils.import_utils import is_package_available
from lerobot.common.policies.act.modeling_act import ActionChunkingTransformerPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.tdmpc.policy import TDMPCPolicy
from tests.utils import require_env


@pytest.mark.parametrize("env_name, task_name", lerobot.env_task_pairs)
@require_env
def test_available_env_task(env_name: str, task_name: list):
    """
    This test verifies that all environments listed in `lerobot/__init__.py` can
    be sucessfully imported — if they're installed — and that their
    `available_tasks_per_env` are valid.
    """
    package_name = f"gym_{env_name}"
    importlib.import_module(package_name)
    gym_handle = f"{package_name}/{task_name}"
    assert gym_handle in gym.envs.registry.keys(), gym_handle


def test_available_policies():
    """
    This test verifies that the class attribute `name` for all policies is
    consistent with those listed in `lerobot/__init__.py`.
    """
    policy_classes = [
        ActionChunkingTransformerPolicy,
        DiffusionPolicy,
        TDMPCPolicy,
    ]
    policies = [pol_cls.name for pol_cls in policy_classes]
    assert set(policies) == set(lerobot.available_policies), policies
