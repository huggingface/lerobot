import pytest
import torch

from lerobot.common.datasets.factory import make_offline_buffer

from .utils import DEVICE, init_config


@pytest.mark.parametrize(
    "env_name,dataset_id",
    [
        # TODO(rcadene): simxarm is depreciated for now
        ("simxarm", "lift"),
        ("pusht", "pusht"),
        ("aloha", "sim_insertion_human"),
        ("aloha", "sim_insertion_scripted"),
        ("aloha", "sim_transfer_cube_human"),
        ("aloha", "sim_transfer_cube_scripted"),
    ],
)
def test_factory(env_name, dataset_id):
    cfg = init_config(overrides=[f"env={env_name}", f"env.task={dataset_id}", f"device={DEVICE}"])
    offline_buffer = make_offline_buffer(cfg)
    for key in offline_buffer.image_keys:
        img = offline_buffer[0].get(key)
        assert img.dtype == torch.float32
        # TODO(rcadene): we assume for now that image normalization takes place in the model
        assert img.max() <= 1.0
        assert img.min() >= 0.0
