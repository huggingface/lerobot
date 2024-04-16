import os
from pathlib import Path
import einops
import pytest
import torch

from lerobot.common.datasets.utils import compute_stats, get_stats_einops_patterns, load_previous_and_future_frames
from lerobot.common.transforms import Prod
from lerobot.common.utils import init_hydra_config
import logging
from lerobot.common.datasets.factory import make_dataset
from datasets import Dataset
from .utils import DEVICE, DEFAULT_CONFIG_PATH


@pytest.mark.parametrize(
    "env_name,dataset_id,policy_name",
    [
        ("xarm", "xarm_lift_medium", "tdmpc"),
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
        ("episode_id", 0, True),
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
        assert dataset.data_dict[key].dtype == torch.uint8, f"{key}"
    
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


def test_compute_stats():
    """Check that the statistics are computed correctly according to the stats_patterns property.

    We compare with taking a straight min, mean, max, std of all the data in one pass (which we can do
    because we are working with a small dataset).
    """
    from lerobot.common.datasets.xarm import XarmDataset

    DATA_DIR = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else None

    # get transform to convert images from uint8 [0,255] to float32 [0,1]
    transform = Prod(in_keys=XarmDataset.image_keys, prod=1 / 255.0)

    dataset = XarmDataset(
        dataset_id="xarm_lift_medium",
        root=DATA_DIR,
        transform=transform,
    )

    # Note: we set the batch size to be smaller than the whole dataset to make sure we are testing batched
    # computation of the statistics. While doing this, we also make sure it works when we don't divide the
    # dataset into even batches. 
    computed_stats = compute_stats(dataset, batch_size=int(len(dataset) * 0.25))

    # get einops patterns to aggregate batches and compute statistics
    stats_patterns = get_stats_einops_patterns(dataset)

    # get all frames from the dataset in the same dtype and range as during compute_stats
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=len(dataset),
        shuffle=False,
    )
    data_dict = next(iter(dataloader))

    # compute stats based on all frames from the dataset without any batching
    expected_stats = {}
    for k, pattern in stats_patterns.items():
        expected_stats[k] = {}
        expected_stats[k]["mean"] = einops.reduce(data_dict[k], pattern, "mean")
        expected_stats[k]["std"] = torch.sqrt(einops.reduce((data_dict[k] - expected_stats[k]["mean"]) ** 2, pattern, "mean"))
        expected_stats[k]["min"] = einops.reduce(data_dict[k], pattern, "min")
        expected_stats[k]["max"] = einops.reduce(data_dict[k], pattern, "max")

    # test computed stats match expected stats
    for k in stats_patterns:
        assert torch.allclose(computed_stats[k]["mean"], expected_stats[k]["mean"])
        assert torch.allclose(computed_stats[k]["std"], expected_stats[k]["std"])
        assert torch.allclose(computed_stats[k]["min"], expected_stats[k]["min"])
        assert torch.allclose(computed_stats[k]["max"], expected_stats[k]["max"])

    # TODO(rcadene): check that the stats used for training are correct too
    # # load stats that are expected to match the ones returned by computed_stats
    # assert (dataset.data_dir / "stats.pth").exists()
    # loaded_stats = torch.load(dataset.data_dir / "stats.pth")

    # # test loaded stats match expected stats
    # for k in stats_patterns:
    #     assert torch.allclose(loaded_stats[k]["mean"], expected_stats[k]["mean"])
    #     assert torch.allclose(loaded_stats[k]["std"], expected_stats[k]["std"])
    #     assert torch.allclose(loaded_stats[k]["min"], expected_stats[k]["min"])
    #     assert torch.allclose(loaded_stats[k]["max"], expected_stats[k]["max"])


def test_load_previous_and_future_frames_within_tolerance():
    data_dict = Dataset.from_dict({
        "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5],
        "index": [0, 1, 2, 3, 4],
        "episode_data_index_from": [0, 0, 0, 0, 0],
        "episode_data_index_to": [4, 4, 4, 4, 4],
    })
    data_dict = data_dict.with_format("torch")
    item = data_dict[2]
    delta_timestamps = {"index": [-0.2, 0, 0.139]}
    tol = 0.04
    item = load_previous_and_future_frames(item, data_dict, delta_timestamps, tol)
    data, is_pad = item["index"], item["index_is_pad"]
    assert torch.equal(data, torch.tensor([0, 2, 3])), "Data does not match expected values"
    assert not is_pad.any(), "Unexpected padding detected"

def test_load_previous_and_future_frames_outside_tolerance_inside_episode_range():
    data_dict = Dataset.from_dict({
        "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5],
        "index": [0, 1, 2, 3, 4],
        "episode_data_index_from": [0, 0, 0, 0, 0],
        "episode_data_index_to": [4, 4, 4, 4, 4],
    })
    data_dict = data_dict.with_format("torch")
    item = data_dict[2]
    delta_timestamps = {"index": [-0.2, 0, 0.141]}
    tol = 0.04
    with pytest.raises(AssertionError):
        load_previous_and_future_frames(item, data_dict, delta_timestamps, tol)

def test_load_previous_and_future_frames_outside_tolerance_outside_episode_range():
    data_dict = Dataset.from_dict({
        "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5],
        "index": [0, 1, 2, 3, 4],
        "episode_data_index_from": [0, 0, 0, 0, 0],
        "episode_data_index_to": [4, 4, 4, 4, 4],
    })
    data_dict = data_dict.with_format("torch")
    item = data_dict[2]
    delta_timestamps = {"index": [-0.3, -0.24, 0, 0.26, 0.3]}
    tol = 0.04
    item = load_previous_and_future_frames(item, data_dict, delta_timestamps, tol)
    data, is_pad = item["index"], item["index_is_pad"]
    assert torch.equal(data, torch.tensor([0, 0, 2, 4, 4])), "Data does not match expected values"
    assert torch.equal(is_pad, torch.tensor([True, False, False, True, True])), "Padding does not match expected values"
    

