import json
import logging
import os
from copy import deepcopy
from pathlib import Path

import einops
import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file

import lerobot
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
)
from lerobot.common.datasets.utils import (
    compute_stats,
    flatten_dict,
    get_stats_einops_patterns,
    hf_transform_to_torch,
    load_previous_and_future_frames,
    unflatten_dict,
)
from lerobot.common.utils.utils import init_hydra_config
from tests.utils import DEFAULT_CONFIG_PATH, DEVICE


@pytest.mark.parametrize("env_name, repo_id, policy_name", lerobot.env_dataset_policy_triplets)
def test_factory(env_name, repo_id, policy_name):
    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[
            f"env={env_name}",
            f"dataset_repo_id={repo_id}",
            f"policy={policy_name}",
            f"device={DEVICE}",
        ],
    )
    dataset = make_dataset(cfg)
    delta_timestamps = dataset.delta_timestamps
    image_keys = dataset.image_keys

    item = dataset[0]

    keys_ndim_required = [
        ("action", 1, True),
        ("episode_index", 0, True),
        ("frame_index", 0, True),
        ("timestamp", 0, True),
        # TODO(rcadene): should we rename it agent_pos?
        ("observation.state", 1, True),
        ("next.reward", 0, False),
        ("next.done", 0, False),
    ]

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


def test_compute_stats_on_xarm():
    """Check that the statistics are computed correctly according to the stats_patterns property.

    We compare with taking a straight min, mean, max, std of all the data in one pass (which we can do
    because we are working with a small dataset).
    """
    dataset = LeRobotDataset(
        "lerobot/xarm_lift_medium", root=Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else None
    )

    # reduce size of dataset sample on which stats compute is tested to 10 frames
    dataset.hf_dataset = dataset.hf_dataset.select(range(10))

    # Note: we set the batch size to be smaller than the whole dataset to make sure we are testing batched
    # computation of the statistics. While doing this, we also make sure it works when we don't divide the
    # dataset into even batches.
    computed_stats = compute_stats(dataset.hf_dataset, batch_size=int(len(dataset) * 0.25))

    # get einops patterns to aggregate batches and compute statistics
    stats_patterns = get_stats_einops_patterns(dataset.hf_dataset)

    # get all frames from the dataset in the same dtype and range as during compute_stats
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=len(dataset),
        shuffle=False,
    )
    full_batch = next(iter(dataloader))

    # compute stats based on all frames from the dataset without any batching
    expected_stats = {}
    for k, pattern in stats_patterns.items():
        full_batch[k] = full_batch[k].float()
        expected_stats[k] = {}
        expected_stats[k]["mean"] = einops.reduce(full_batch[k], pattern, "mean")
        expected_stats[k]["std"] = torch.sqrt(
            einops.reduce((full_batch[k] - expected_stats[k]["mean"]) ** 2, pattern, "mean")
        )
        expected_stats[k]["min"] = einops.reduce(full_batch[k], pattern, "min")
        expected_stats[k]["max"] = einops.reduce(full_batch[k], pattern, "max")

    # test computed stats match expected stats
    for k in stats_patterns:
        assert torch.allclose(computed_stats[k]["mean"], expected_stats[k]["mean"])
        assert torch.allclose(computed_stats[k]["std"], expected_stats[k]["std"])
        assert torch.allclose(computed_stats[k]["min"], expected_stats[k]["min"])
        assert torch.allclose(computed_stats[k]["max"], expected_stats[k]["max"])

    # load stats used during training which are expected to match the ones returned by computed_stats
    loaded_stats = dataset.stats  # noqa: F841

    # TODO(rcadene): we can't test this because expected_stats is computed on a subset
    # # test loaded stats match expected stats
    # for k in stats_patterns:
    #     assert torch.allclose(loaded_stats[k]["mean"], expected_stats[k]["mean"])
    #     assert torch.allclose(loaded_stats[k]["std"], expected_stats[k]["std"])
    #     assert torch.allclose(loaded_stats[k]["min"], expected_stats[k]["min"])
    #     assert torch.allclose(loaded_stats[k]["max"], expected_stats[k]["max"])


def test_load_previous_and_future_frames_within_tolerance():
    hf_dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5],
            "index": [0, 1, 2, 3, 4],
            "episode_index": [0, 0, 0, 0, 0],
        }
    )
    hf_dataset.set_transform(hf_transform_to_torch)
    episode_data_index = {
        "from": torch.tensor([0]),
        "to": torch.tensor([5]),
    }
    delta_timestamps = {"index": [-0.2, 0, 0.139]}
    tol = 0.04
    item = hf_dataset[2]
    item = load_previous_and_future_frames(item, hf_dataset, episode_data_index, delta_timestamps, tol)
    data, is_pad = item["index"], item["index_is_pad"]
    assert torch.equal(data, torch.tensor([0, 2, 3])), "Data does not match expected values"
    assert not is_pad.any(), "Unexpected padding detected"


def test_load_previous_and_future_frames_outside_tolerance_inside_episode_range():
    hf_dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5],
            "index": [0, 1, 2, 3, 4],
            "episode_index": [0, 0, 0, 0, 0],
        }
    )
    hf_dataset.set_transform(hf_transform_to_torch)
    episode_data_index = {
        "from": torch.tensor([0]),
        "to": torch.tensor([5]),
    }
    delta_timestamps = {"index": [-0.2, 0, 0.141]}
    tol = 0.04
    item = hf_dataset[2]
    with pytest.raises(AssertionError):
        load_previous_and_future_frames(item, hf_dataset, episode_data_index, delta_timestamps, tol)


def test_load_previous_and_future_frames_outside_tolerance_outside_episode_range():
    hf_dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5],
            "index": [0, 1, 2, 3, 4],
            "episode_index": [0, 0, 0, 0, 0],
        }
    )
    hf_dataset.set_transform(hf_transform_to_torch)
    episode_data_index = {
        "from": torch.tensor([0]),
        "to": torch.tensor([5]),
    }
    delta_timestamps = {"index": [-0.3, -0.24, 0, 0.26, 0.3]}
    tol = 0.04
    item = hf_dataset[2]
    item = load_previous_and_future_frames(item, hf_dataset, episode_data_index, delta_timestamps, tol)
    data, is_pad = item["index"], item["index_is_pad"]
    assert torch.equal(data, torch.tensor([0, 0, 2, 4, 4])), "Data does not match expected values"
    assert torch.equal(
        is_pad, torch.tensor([True, False, False, True, True])
    ), "Padding does not match expected values"


def test_flatten_unflatten_dict():
    d = {
        "obs": {
            "min": 0,
            "max": 1,
            "mean": 2,
            "std": 3,
        },
        "action": {
            "min": 4,
            "max": 5,
            "mean": 6,
            "std": 7,
        },
    }

    original_d = deepcopy(d)
    d = unflatten_dict(flatten_dict(d))

    # test equality between nested dicts
    assert json.dumps(original_d, sort_keys=True) == json.dumps(d, sort_keys=True), f"{original_d} != {d}"


@pytest.mark.parametrize(
    "repo_id",
    [
        "lerobot/pusht",
        "lerobot/aloha_sim_insertion_human",
        "lerobot/xarm_lift_medium",
    ],
)
def test_backward_compatibility(repo_id):
    """The artifacts for this test have been generated by `tests/scripts/save_dataset_to_safetensors.py`."""

    dataset = LeRobotDataset(
        repo_id,
        root=Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else None,
    )

    test_dir = Path("tests/data/save_dataset_to_safetensors") / repo_id

    def load_and_compare(i):
        new_frame = dataset[i]  # noqa: B023
        old_frame = load_file(test_dir / f"frame_{i}.safetensors")  # noqa: B023

        new_keys = set(new_frame.keys())
        old_keys = set(old_frame.keys())
        assert new_keys == old_keys, f"{new_keys=} and {old_keys=} are not the same"

        for key in new_frame:
            assert torch.isclose(
                new_frame[key], old_frame[key], rtol=1e-05, atol=1e-08
            ).all(), f"{key=} for index={i} does not contain the same value"

    # test2 first frames of first episode
    i = dataset.episode_data_index["from"][0].item()
    load_and_compare(i)
    load_and_compare(i + 1)

    # test 2 frames at the middle of first episode
    i = int((dataset.episode_data_index["to"][0].item() - dataset.episode_data_index["from"][0].item()) / 2)
    load_and_compare(i)
    load_and_compare(i + 1)

    # test 2 last frames of first episode
    i = dataset.episode_data_index["to"][0].item()
    load_and_compare(i - 2)
    load_and_compare(i - 1)

    # TODO(rcadene): Enable testing on second and last episode
    # We currently cant because our test dataset only contains the first episode

    # # test 2 first frames of second episode
    # i = dataset.episode_data_index["from"][1].item()
    # load_and_compare(i)
    # load_and_compare(i + 1)

    # # test 2 last frames of second episode
    # i = dataset.episode_data_index["to"][1].item()
    # load_and_compare(i - 2)
    # load_and_compare(i - 1)

    # # test 2 last frames of last episode
    # i = dataset.episode_data_index["to"][-1].item()
    # load_and_compare(i - 2)
    # load_and_compare(i - 1)
