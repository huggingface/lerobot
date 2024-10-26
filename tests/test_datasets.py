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

import pytest
import torch
from huggingface_hub import HfApi
from safetensors.torch import load_file

import lerobot
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    MultiLeRobotDataset,
)
from lerobot.common.datasets.utils import (
    create_branch,
    flatten_dict,
    unflatten_dict,
)
from lerobot.common.envs.factory import make_env_config
from lerobot.common.policies.factory import make_policy_config
from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.utils import DEVICE, require_x86_64_kernel


def test_same_attributes_defined(lerobot_dataset_factory, tmp_path):
    """
    Instantiate a LeRobotDataset both ways with '__init__()' and 'create()' and verify that instantiated
    objects have the same sets of attributes defined.
    """
    # Instantiate both ways
    robot = make_robot("koch", mock=True)
    root_create = tmp_path / "create"
    dataset_create = LeRobotDataset.create(repo_id=DUMMY_REPO_ID, fps=30, robot=robot, root=root_create)

    root_init = tmp_path / "init"
    dataset_init = lerobot_dataset_factory(root=root_init)

    # Access the '_hub_version' cached_property in both instances to force its creation
    _ = dataset_init.meta._hub_version
    _ = dataset_create.meta._hub_version

    init_attr = set(vars(dataset_init).keys())
    create_attr = set(vars(dataset_create).keys())

    assert init_attr == create_attr


def test_dataset_initialization(lerobot_dataset_factory, tmp_path):
    kwargs = {
        "repo_id": DUMMY_REPO_ID,
        "total_episodes": 10,
        "total_frames": 400,
        "episodes": [2, 5, 6],
    }
    dataset = lerobot_dataset_factory(root=tmp_path, **kwargs)

    assert dataset.repo_id == kwargs["repo_id"]
    assert dataset.meta.total_episodes == kwargs["total_episodes"]
    assert dataset.meta.total_frames == kwargs["total_frames"]
    assert dataset.episodes == kwargs["episodes"]
    assert dataset.num_episodes == len(kwargs["episodes"])
    assert dataset.num_frames == len(dataset)


# TODO(aliberts):
# - [ ] test various attributes & state from init and create
# - [ ] test init with episodes and check num_frames
# - [ ] test add_frame
# - [ ] test add_episode
# - [ ] test consolidate
# - [ ] test push_to_hub
# - [ ] test smaller methods


@pytest.mark.parametrize(
    "env_name, repo_id, policy_name",
    # Single dataset
    lerobot.env_dataset_policy_triplets,
    # Multi-dataset
    # TODO after fix multidataset
    # + [("aloha", ["lerobot/aloha_sim_insertion_human", "lerobot/aloha_sim_transfer_cube_human"], "act")],
)
def test_factory(env_name, repo_id, policy_name):
    """
    Tests that:
        - we can create a dataset with the factory.
        - for a commonly used set of data keys, the data dimensions are correct.
    """
    cfg = TrainPipelineConfig(
        # TODO(rcadene, aliberts): remove dataset download
        dataset=DatasetConfig(repo_id=repo_id, episodes=[0]),
        env=make_env_config(env_name),
        policy=make_policy_config(policy_name),
        device=DEVICE,
    )

    dataset = make_dataset(cfg)
    delta_timestamps = dataset.delta_timestamps
    camera_keys = dataset.meta.camera_keys

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
@pytest.mark.skip("TODO after fix multidataset")
def test_multidataset_frames():
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
    assert dataset.num_frames == sum(d.num_frames for d in sub_datasets)
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


# TODO(aliberts): Move to more appropriate location
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
@require_x86_64_kernel
def test_backward_compatibility(repo_id):
    """The artifacts for this test have been generated by `tests/scripts/save_dataset_to_safetensors.py`."""

    # TODO(rcadene, aliberts): remove dataset download
    dataset = LeRobotDataset(repo_id, episodes=[0])

    test_dir = Path("tests/data/save_dataset_to_safetensors") / repo_id

    def load_and_compare(i):
        new_frame = dataset[i]  # noqa: B023
        old_frame = load_file(test_dir / f"frame_{i}.safetensors")  # noqa: B023

        # ignore language instructions (if exists) in language conditioned datasets
        # TODO (michel-aractingi): transform language obs to langauge embeddings via tokenizer
        new_frame.pop("language_instruction", None)
        old_frame.pop("language_instruction", None)

        # Remove task_index to allow for backward compatibility
        # TODO(rcadene): remove when new features have been generated
        if "task_index" not in old_frame:
            del new_frame["task_index"]

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
