import pytest

from lerobot.common.datasets.factory import make_offline_buffer

from .utils import init_config


@pytest.mark.parametrize(
    "env_name,dataset_id",
    [
        # TODO(rcadene): simxarm is depreciated for now
        # ("simxarm", "lift"),
        ("pusht", "pusht"),
        # TODO(aliberts): add aloha when dataset is available on hub
        # ("aloha", "sim_insertion_human"),
        # ("aloha", "sim_insertion_scripted"),
        # ("aloha", "sim_transfer_cube_human"),
        # ("aloha", "sim_transfer_cube_scripted"),
    ],
)
def test_factory(env_name, dataset_id):
    cfg = init_config(overrides=[f"env={env_name}", f"env.task={dataset_id}"])
    offline_buffer = make_offline_buffer(cfg)
