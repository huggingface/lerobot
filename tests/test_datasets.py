import pytest
import torch

from lerobot.common.utils import init_hydra_config
import logging
from lerobot.common.datasets.factory import make_dataset

from .utils import DEVICE, DEFAULT_CONFIG_PATH


@pytest.mark.parametrize(
    "env_name,dataset_id,policy_name",
    [
        ("simxarm", "xarm_lift_medium", "tdmpc"),
        ("pusht", "pusht", "diffusion"),
        ("aloha", "aloha_sim_insertion_human", "act"),
        ("aloha", "aloha_sim_insertion_scripted", "act"),
        ("aloha", "aloha_sim_transfer_cube_human", "act"),
        ("aloha", "aloha_sim_transfer_cube_scripted", "act"),
    ],
)
def test_factory(env_name, dataset_id, policy_name):
    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[f"env={env_name}", f"dataset_id={dataset_id}", f"policy={policy_name}", f"device={DEVICE}"]
    )
    dataset = make_dataset(cfg)
    delta_timestamps = dataset.delta_timestamps
    image_keys = dataset.image_keys

    item = dataset[0]

    keys_ndim_required = [
        ("action", 1, True),
        ("episode", 0, True),
        ("frame_id", 0, True),
        ("timestamp", 0, True),
        # TODO(rcadene): should we rename it agent_pos?
        ("observation.state", 1, True),
        ("next.reward", 0, False),
        ("next.done", 0, False),
    ]

    for key in image_keys:
        keys_ndim_required.append(
            (key, 3, True),
        )
    
    # test number of dimensions
    for key, ndim, required in keys_ndim_required:
        if key not in item:
            if required:
                assert key in item, f"{key}"
            else:
                logging.warning(f'Missing key in dataset: "{key}" not in {dataset}.')
                continue
        
        if delta_timestamps is not None and key in delta_timestamps:
            assert item[key].ndim == ndim + 1, f"{key}"
            assert item[key].shape[0] == len(delta_timestamps[key]), f"{key}"
        else:
            assert item[key].ndim == ndim, f"{key}"
        
        if key in image_keys:
            assert item[key].dtype == torch.float32, f"{key}"
            # TODO(rcadene): we assume for now that image normalization takes place in the model
            assert item[key].max() <= 1.0, f"{key}"
            assert item[key].min() >= 0.0, f"{key}"

            if delta_timestamps is not None and key in delta_timestamps:
                # test t,c,h,w
                assert item[key].shape[1] == 3, f"{key}"
            else:
                # test c,h,w 
                assert item[key].shape[0] == 3, f"{key}"


    if delta_timestamps is not None:
        # test missing keys in delta_timestamps
        for key in delta_timestamps:
            assert key in item, f"{key}"


# def test_compute_stats():
#     """Check that the statistics are computed correctly according to the stats_patterns property.

#     We compare with taking a straight min, mean, max, std of all the data in one pass (which we can do
#     because we are working with a small dataset).
#     """
#     cfg = init_hydra_config(
#         DEFAULT_CONFIG_PATH, overrides=["env=aloha", "env.task=sim_transfer_cube_human"]
#     )
#     dataset = make_dataset(cfg)
#     # Get all of the data.
#     all_data = dataset.data_dict
#     # Note: we set the batch size to be smaller than the whole dataset to make sure we are testing batched
#     # computation of the statistics. While doing this, we also make sure it works when we don't divide the
#     # dataset into even batches. 
#     computed_stats = buffer._compute_stats(batch_size=int(len(all_data) * 0.75))
#     for k, pattern in buffer.stats_patterns.items():
#         expected_mean = einops.reduce(all_data[k], pattern, "mean")
#         assert torch.allclose(computed_stats[k]["mean"], expected_mean)
#         assert torch.allclose(
#             computed_stats[k]["std"],
#             torch.sqrt(einops.reduce((all_data[k] - expected_mean) ** 2, pattern, "mean"))
#         )
#         assert torch.allclose(computed_stats[k]["min"], einops.reduce(all_data[k], pattern, "min"))
#         assert torch.allclose(computed_stats[k]["max"], einops.reduce(all_data[k], pattern, "max"))
