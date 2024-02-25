import pytest

from lerobot.common.datasets.factory import make_offline_buffer

from .utils import init_config


@pytest.mark.parametrize(
    "env_name",
    [
        "simxarm",
        "pusht",
    ],
)
def test_factory(env_name):
    cfg = init_config(overrides=[f"env={env_name}"])
    offline_buffer = make_offline_buffer(cfg)
