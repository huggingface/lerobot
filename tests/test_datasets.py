import einops
import pytest
import torch
from torchrl.data.replay_buffers.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from lerobot.common.datasets.factory import make_offline_buffer
from lerobot.common.utils import init_hydra_config

from .utils import DEVICE, DEFAULT_CONFIG_PATH


@pytest.mark.parametrize(
    "env_name,dataset_id",
    [
        ("simxarm", "lift"),
        ("pusht", "pusht"),
        ("aloha", "sim_insertion_human"),
        ("aloha", "sim_insertion_scripted"),
        ("aloha", "sim_transfer_cube_human"),
        ("aloha", "sim_transfer_cube_scripted"),
    ],
)
def test_factory(env_name, dataset_id):
    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[f"env={env_name}", f"env.task={dataset_id}", f"device={DEVICE}"]
    )
    offline_buffer = make_offline_buffer(cfg)
    for key in offline_buffer.image_keys:
        img = offline_buffer[0].get(key)
        assert img.dtype == torch.float32
        # TODO(rcadene): we assume for now that image normalization takes place in the model
        assert img.max() <= 1.0
        assert img.min() >= 0.0


def test_compute_stats():
    """Check that the statistics are computed correctly according to the stats_patterns property.

    We compare with taking a straight min, mean, max, std of all the data in one pass (which we can do
    because we are working with a small dataset).
    """
    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH, overrides=["env=aloha", "env.task=sim_transfer_cube_human"]
    )
    buffer = make_offline_buffer(cfg)
    # Get all of the data.
    all_data = TensorDictReplayBuffer(
            storage=buffer._storage,
            batch_size=len(buffer),
            sampler=SamplerWithoutReplacement(),
    ).sample().float()
    # Note: we set the batch size to be smaller than the whole dataset to make sure we are testing batched
    # computation of the statistics. While doing this, we also make sure it works when we don't divide the
    # dataset into even batches. 
    computed_stats = buffer._compute_stats(batch_size=int(len(all_data) * 0.75))
    for k, pattern in buffer.stats_patterns.items():
        expected_mean = einops.reduce(all_data[k], pattern, "mean")
        assert torch.allclose(computed_stats[k]["mean"], expected_mean)
        try:
            assert torch.allclose(
                computed_stats[k]["std"],
                torch.sqrt(einops.reduce((all_data[k] - expected_mean) ** 2, pattern, "mean"))
            )
        except:
            breakpoint()
        assert torch.allclose(computed_stats[k]["min"], einops.reduce(all_data[k], pattern, "min"))
        assert torch.allclose(computed_stats[k]["max"], einops.reduce(all_data[k], pattern, "max"))
