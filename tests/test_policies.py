import pytest

from lerobot.common.policies.factory import make_policy

from .utils import init_config


@pytest.mark.parametrize(
    "config_name",
    [
        "default",
        "pusht",
    ],
)
def test_factory(config_name):
    cfg = init_config(config_name)
    policy = make_policy(cfg)
