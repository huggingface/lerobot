#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
from copy import deepcopy
from itertools import chain
from pathlib import Path

import einops
import pytest
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from safetensors.torch import load_file

import lerobot
from lerobot.common.datasets.compute_stats import (
    aggregate_stats,
    compute_stats,
    get_stats_einops_patterns,
)
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.common.datasets.utils import (
    create_branch,
    flatten_dict,
    hf_transform_to_torch,
    load_previous_and_future_frames,
    unflatten_dict,
)
from lerobot.common.utils.utils import init_hydra_config, seeded_context
from tests.utils import DEFAULT_CONFIG_PATH, DEVICE


@pytest.mark.parametrize(
    "env_name, repo_id, policy_name",
    lerobot.env_dataset_policy_triplets
    + [("aloha", ["lerobot/aloha_sim_insertion_human", "lerobot/aloha_sim_transfer_cube_human"], "act")],
)
def test_factory(env_name, repo_id, policy_name):
    """
    Tests that:
        - we can create a dataset with the factory.
        - for a commonly used set of data keys, the data dimensions are correct.
    """
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
    camera_keys = dataset.camera_keys

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

        if key in camera_keys:
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


# TODO(alexander-soare): If you're hunting for savings on testing time, this takes about 5 seconds.
def test_multilerobotdataset_frames():
    """Check that all dataset frames are incorporated."""
    # Note: use the image variants of the dataset to make the test approx 3x faster.
    # Note: We really do need three repo_ids here as at some point this caught an issue with the chaining
    # logic that wouldn't be caught with two repo IDs.
    repo_ids = [
        "lerobot/aloha_sim_insertion_human_image",
        "lerobot/aloha_sim_transfer_cube_human_image",
        "lerobot/aloha_sim_insertion_scripted_image",
    ]
    sub_datasets = [LeRobotDataset(repo_id) for repo_id in repo_ids]
    dataset = MultiLeRobotDataset(repo_ids)
    assert len(dataset) == sum(len(d) for d in sub_datasets)
    assert dataset.num_samples == sum(d.num_samples for d in sub_datasets)
    assert dataset.num_episodes == sum(d.num_episodes for d in sub_datasets)

    # Run through all items of the LeRobotDatasets in parallel with the items of the MultiLerobotDataset and
    # check they match.
    expected_dataset_indices = []
    for i, sub_dataset in enumerate(sub_datasets):
        expected_dataset_indices.extend([i] * len(sub_dataset))

    for expected_dataset_index, sub_dataset_item, dataset_item in zip(
        expected_dataset_indices, chain(*sub_datasets), dataset, strict=True
    ):
        dataset_index = dataset_item.pop("dataset_index")
        assert dataset_index == expected_dataset_index
        assert sub_dataset_item.keys() == dataset_item.keys()
        for k in sub_dataset_item:
            assert torch.equal(sub_dataset_item[k], dataset_item[k])


def test_compute_stats_on_xarm():
    """Check that the statistics are computed correctly according to the stats_patterns property.

    We compare with taking a straight min, mean, max, std of all the data in one pass (which we can do
    because we are working with a small dataset).
    """
    dataset = LeRobotDataset("lerobot/xarm_lift_medium")

    # reduce size of dataset sample on which stats compute is tested to 10 frames
    dataset.hf_dataset = dataset.hf_dataset.select(range(10))

    # Note: we set the batch size to be smaller than the whole dataset to make sure we are testing batched
    # computation of the statistics. While doing this, we also make sure it works when we don't divide the
    # dataset into even batches.
    computed_stats = compute_stats(dataset, batch_size=int(len(dataset) * 0.25), num_workers=0)

    # get einops patterns to aggregate batches and compute statistics
    stats_patterns = get_stats_einops_patterns(dataset)

    # get all frames from the dataset in the same dtype and range as during compute_stats
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
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
        # (michel-aractingi) commenting the two datasets from openx as test is failing
        # "lerobot/nyu_franka_play_dataset",
        # "lerobot/cmu_stretch",
    ],
)
# TODO(rcadene, aliberts): all these tests fail locally on Mac M1, but not on Linux
def test_backward_compatibility(repo_id):
    """The artifacts for this test have been generated by `tests/scripts/save_dataset_to_safetensors.py`."""

    dataset = LeRobotDataset(repo_id)

    test_dir = Path("tests/data/save_dataset_to_safetensors") / repo_id

    def load_and_compare(i):
        new_frame = dataset[i]  # noqa: B023
        old_frame = load_file(test_dir / f"frame_{i}.safetensors")  # noqa: B023

        # ignore language instructions (if exists) in language conditioned datasets
        # TODO (michel-aractingi): transform language obs to langauge embeddings via tokenizer
        new_frame.pop("language_instruction", None)
        old_frame.pop("language_instruction", None)

        new_keys = set(new_frame.keys())
        old_keys = set(old_frame.keys())
        assert new_keys == old_keys, f"{new_keys=} and {old_keys=} are not the same"

        for key in new_frame:
            assert torch.isclose(
                new_frame[key], old_frame[key]
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


def test_aggregate_stats():
    """Makes 3 basic datasets and checks that aggregate stats are computed correctly."""
    with seeded_context(0):
        data_a = torch.rand(30, dtype=torch.float32)
        data_b = torch.rand(20, dtype=torch.float32)
        data_c = torch.rand(20, dtype=torch.float32)

    hf_dataset_1 = Dataset.from_dict(
        {"a": data_a[:10], "b": data_b[:10], "c": data_c[:10], "index": torch.arange(10)}
    )
    hf_dataset_1.set_transform(hf_transform_to_torch)
    hf_dataset_2 = Dataset.from_dict({"a": data_a[10:20], "b": data_b[10:], "index": torch.arange(10)})
    hf_dataset_2.set_transform(hf_transform_to_torch)
    hf_dataset_3 = Dataset.from_dict({"a": data_a[20:], "c": data_c[10:], "index": torch.arange(10)})
    hf_dataset_3.set_transform(hf_transform_to_torch)
    dataset_1 = LeRobotDataset.from_preloaded("d1", hf_dataset=hf_dataset_1)
    dataset_1.stats = compute_stats(dataset_1, batch_size=len(hf_dataset_1), num_workers=0)
    dataset_2 = LeRobotDataset.from_preloaded("d2", hf_dataset=hf_dataset_2)
    dataset_2.stats = compute_stats(dataset_2, batch_size=len(hf_dataset_2), num_workers=0)
    dataset_3 = LeRobotDataset.from_preloaded("d3", hf_dataset=hf_dataset_3)
    dataset_3.stats = compute_stats(dataset_3, batch_size=len(hf_dataset_3), num_workers=0)
    stats = aggregate_stats([dataset_1, dataset_2, dataset_3])
    for data_key, data in zip(["a", "b", "c"], [data_a, data_b, data_c], strict=True):
        for agg_fn in ["mean", "min", "max"]:
            assert torch.allclose(stats[data_key][agg_fn], einops.reduce(data, "n -> 1", agg_fn))
        assert torch.allclose(stats[data_key]["std"], torch.std(data, correction=0))


@pytest.mark.skip("Requires internet access")
def test_create_branch():
    api = HfApi()

    repo_id = "cadene/test_create_branch"
    repo_type = "dataset"
    branch = "test"
    ref = f"refs/heads/{branch}"

    # Prepare a repo with a test branch
    api.delete_repo(repo_id, repo_type=repo_type, missing_ok=True)
    api.create_repo(repo_id, repo_type=repo_type)
    create_branch(repo_id, repo_type=repo_type, branch=branch)

    # Make sure the test branch exists
    branches = api.list_repo_refs(repo_id, repo_type=repo_type).branches
    refs = [branch.ref for branch in branches]
    assert ref in refs

    # Overwrite it
    create_branch(repo_id, repo_type=repo_type, branch=branch)

    # Clean
    api.delete_repo(repo_id, repo_type=repo_type)
