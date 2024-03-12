import pytest

from lerobot.common.policies.factory import make_policy

from .utils import DEVICE, init_config


@pytest.mark.parametrize(
    "env_name,policy_name",
    [
        ("simxarm", "tdmpc"),
        ("pusht", "tdmpc"),
        ("simxarm", "diffusion"),
        ("pusht", "diffusion"),
    ],
)
def test_factory(env_name, policy_name):
    cfg = init_config(
        overrides=[
            f"env={env_name}",
            f"policy={policy_name}",
            f"device={DEVICE}",
        ]
    )
    policy = make_policy(cfg)
