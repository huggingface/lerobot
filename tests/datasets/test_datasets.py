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
import logging
import re
from itertools import chain
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import HfApi
from PIL import Image
from safetensors.torch import load_file

import lerobot
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.image_writer import image_array_to_pil_image
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    MultiLeRobotDataset,
)
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    create_branch,
    get_hf_features_from_features,
    hf_transform_to_torch,
    hw_to_dataset_features,
)
from lerobot.envs.factory import make_env_config
from lerobot.policies.factory import make_policy_config
from lerobot.robots import make_robot_from_config
from tests.fixtures.constants import DUMMY_CHW, DUMMY_HWC, DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobotConfig
from tests.utils import require_x86_64_kernel


@pytest.fixture
def image_dataset(tmp_path, empty_lerobot_dataset_factory):
    features = {
        "image": {
            "dtype": "image",
            "shape": DUMMY_CHW,
            "names": [
                "channels",
                "height",
                "width",
            ],
        }
    }
    return empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)


def test_same_attributes_defined(tmp_path, lerobot_dataset_factory):
    """
    Instantiate a LeRobotDataset both ways with '__init__()' and 'create()' and verify that instantiated
    objects have the same sets of attributes defined.
    """
    # Instantiate both ways
    robot = make_robot_from_config(MockRobotConfig())
    action_features = hw_to_dataset_features(robot.action_features, "action", True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)
    dataset_features = {**action_features, **obs_features}
    root_create = tmp_path / "create"
    dataset_create = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=30, features=dataset_features, root=root_create
    )

    root_init = tmp_path / "init"
    dataset_init = lerobot_dataset_factory(root=root_init, total_episodes=1, total_frames=1)

    init_attr = set(vars(dataset_init).keys())
    create_attr = set(vars(dataset_create).keys())

    assert init_attr == create_attr


def test_dataset_initialization(tmp_path, lerobot_dataset_factory):
    kwargs = {
        "repo_id": DUMMY_REPO_ID,
        "total_episodes": 10,
        "total_frames": 400,
        "episodes": [2, 5, 6],
    }
    dataset = lerobot_dataset_factory(root=tmp_path / "test", **kwargs)

    assert dataset.repo_id == kwargs["repo_id"]
    assert dataset.meta.total_episodes == kwargs["total_episodes"]
    assert dataset.meta.total_frames == kwargs["total_frames"]
    assert dataset.episodes == kwargs["episodes"]
    assert dataset.num_episodes == len(kwargs["episodes"])
    assert dataset.num_frames == len(dataset)


# TODO(rcadene, aliberts): do not run LeRobotDataset.create, instead refactor LeRobotDatasetMetadata.create
# and test the small resulting function that validates the features
def test_dataset_feature_with_forward_slash_raises_error():
    # make sure dir does not exist
    from lerobot.constants import HF_LEROBOT_HOME

    dataset_dir = HF_LEROBOT_HOME / "lerobot/test/with/slash"
    # make sure does not exist
    if dataset_dir.exists():
        dataset_dir.rmdir()

    with pytest.raises(ValueError):
        LeRobotDataset.create(
            repo_id="lerobot/test/with/slash",
            fps=30,
            features={"a/b": {"dtype": "float32", "shape": 2, "names": None}},
        )


def test_add_frame_missing_task(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError, match="Feature mismatch in `frame` dictionary:\nMissing features: {'task'}\n"
    ):
        dataset.add_frame({"state": torch.randn(1)})


def test_add_frame_missing_feature(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError, match="Feature mismatch in `frame` dictionary:\nMissing features: {'state'}\n"
    ):
        dataset.add_frame({"task": "Dummy task"})


def test_add_frame_extra_feature(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError, match="Feature mismatch in `frame` dictionary:\nExtra features: {'extra'}\n"
    ):
        dataset.add_frame({"state": torch.randn(1), "task": "Dummy task", "extra": "dummy_extra"})


def test_add_frame_wrong_type(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError, match="The feature 'state' of dtype 'float16' is not of the expected dtype 'float32'.\n"
    ):
        dataset.add_frame({"state": torch.randn(1, dtype=torch.float16), "task": "Dummy task"})


def test_add_frame_wrong_shape(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError,
        match=re.escape("The feature 'state' of shape '(1,)' does not have the expected shape '(2,)'.\n"),
    ):
        dataset.add_frame({"state": torch.randn(1), "task": "Dummy task"})


def test_add_frame_wrong_shape_python_float(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The feature 'state' is not a 'np.ndarray'. Expected type is 'float32', but type '<class 'float'>' provided instead.\n"
        ),
    ):
        dataset.add_frame({"state": 1.0, "task": "Dummy task"})


def test_add_frame_wrong_shape_torch_ndim_0(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError,
        match=re.escape("The feature 'state' of shape '()' does not have the expected shape '(1,)'.\n"),
    ):
        dataset.add_frame({"state": torch.tensor(1.0), "task": "Dummy task"})


def test_add_frame_wrong_shape_numpy_ndim_0(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The feature 'state' is not a 'np.ndarray'. Expected type is 'float32', but type '<class 'numpy.float32'>' provided instead.\n"
        ),
    ):
        dataset.add_frame({"state": np.float32(1.0), "task": "Dummy task"})


def test_add_frame(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    dataset.add_frame({"state": torch.randn(1), "task": "Dummy task"})
    dataset.save_episode()

    assert len(dataset) == 1
    assert dataset[0]["task"] == "Dummy task"
    assert dataset[0]["task_index"] == 0
    assert dataset[0]["state"].ndim == 0


def test_add_frame_state_1d(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    dataset.add_frame({"state": torch.randn(2), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["state"].shape == torch.Size([2])


def test_add_frame_state_2d(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (2, 4), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    dataset.add_frame({"state": torch.randn(2, 4), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["state"].shape == torch.Size([2, 4])


def test_add_frame_state_3d(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (2, 4, 3), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    dataset.add_frame({"state": torch.randn(2, 4, 3), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["state"].shape == torch.Size([2, 4, 3])


def test_add_frame_state_4d(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (2, 4, 3, 5), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    dataset.add_frame({"state": torch.randn(2, 4, 3, 5), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["state"].shape == torch.Size([2, 4, 3, 5])


def test_add_frame_state_5d(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (2, 4, 3, 5, 1), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    dataset.add_frame({"state": torch.randn(2, 4, 3, 5, 1), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["state"].shape == torch.Size([2, 4, 3, 5, 1])


def test_add_frame_state_numpy(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    dataset.add_frame({"state": np.array([1], dtype=np.float32), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["state"].ndim == 0


def test_add_frame_string(tmp_path, empty_lerobot_dataset_factory):
    features = {"caption": {"dtype": "string", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    dataset.add_frame({"caption": "Dummy caption", "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["caption"] == "Dummy caption"


def test_add_frame_image_wrong_shape(image_dataset):
    dataset = image_dataset
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The feature 'image' of shape '(3, 128, 96)' does not have the expected shape '(3, 96, 128)' or '(96, 128, 3)'.\n"
        ),
    ):
        c, h, w = DUMMY_CHW
        dataset.add_frame({"image": torch.randn(c, w, h), "task": "Dummy task"})


def test_add_frame_image_wrong_range(image_dataset):
    """This test will display the following error message from a thread:
    ```
    Error writing image ...test_add_frame_image_wrong_ran0/test/images/image/episode_000000/frame_000000.png:
    The image data type is float, which requires values in the range [0.0, 1.0]. However, the provided range is [0.009678772038470007, 254.9776492089887].
    Please adjust the range or provide a uint8 image with values in the range [0, 255]
    ```
    Hence the image won't be saved on disk and save_episode will raise `FileNotFoundError`.
    """
    dataset = image_dataset
    dataset.add_frame({"image": np.random.rand(*DUMMY_CHW) * 255, "task": "Dummy task"})
    with pytest.raises(FileNotFoundError):
        dataset.save_episode()


def test_add_frame_image(image_dataset):
    dataset = image_dataset
    dataset.add_frame({"image": np.random.rand(*DUMMY_CHW), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["image"].shape == torch.Size(DUMMY_CHW)


def test_add_frame_image_h_w_c(image_dataset):
    dataset = image_dataset
    dataset.add_frame({"image": np.random.rand(*DUMMY_HWC), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["image"].shape == torch.Size(DUMMY_CHW)


def test_add_frame_image_uint8(image_dataset):
    dataset = image_dataset
    image = np.random.randint(0, 256, DUMMY_HWC, dtype=np.uint8)
    dataset.add_frame({"image": image, "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["image"].shape == torch.Size(DUMMY_CHW)


def test_add_frame_image_pil(image_dataset):
    dataset = image_dataset
    image = np.random.randint(0, 256, DUMMY_HWC, dtype=np.uint8)
    dataset.add_frame({"image": Image.fromarray(image), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["image"].shape == torch.Size(DUMMY_CHW)


def test_image_array_to_pil_image_wrong_range_float_0_255():
    image = np.random.rand(*DUMMY_HWC) * 255
    with pytest.raises(ValueError):
        image_array_to_pil_image(image)


# TODO(aliberts):
# - [ ] test various attributes & state from init and create
# - [ ] test init with episodes and check num_frames
# - [ ] test add_episode
# - [ ] test push_to_hub
# - [ ] test smaller methods

# TODO(rcadene):
# - [ ] fix code so that old test_factory + backward pass
# - [ ] write new unit tests to test save_episode + getitem
#   - [ ] save_episode : case where new dataset, concatenate same file, write new file (meta/episodes, data, videos)
#   - [ ]
# - [ ] remove old tests


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
    """The artifacts for this test have been generated by `tests/artifacts/datasets/save_dataset_to_safetensors.py`."""

    # TODO(rcadene, aliberts): remove dataset download
    dataset = LeRobotDataset(repo_id, episodes=[0])

    test_dir = Path("tests/artifacts/datasets") / repo_id

    def load_and_compare(i):
        new_frame = dataset[i]  # noqa: B023
        old_frame = load_file(test_dir / f"frame_{i}.safetensors")  # noqa: B023

        # ignore language instructions (if exists) in language conditioned datasets
        # TODO (michel-aractingi): transform language obs to language embeddings via tokenizer
        new_frame.pop("language_instruction", None)
        old_frame.pop("language_instruction", None)
        new_frame.pop("task", None)
        old_frame.pop("task", None)

        # Remove task_index to allow for backward compatibility
        # TODO(rcadene): remove when new features have been generated
        if "task_index" not in old_frame:
            del new_frame["task_index"]

        new_keys = set(new_frame.keys())
        old_keys = set(old_frame.keys())
        assert new_keys == old_keys, f"{new_keys=} and {old_keys=} are not the same"

        for key in new_frame:
            assert torch.isclose(new_frame[key], old_frame[key]).all(), (
                f"{key=} for index={i} does not contain the same value"
            )

    # test2 first frames of first episode
    i = dataset.meta.episodes[0]["dataset_from_index"]
    load_and_compare(i)
    load_and_compare(i + 1)

    # test 2 frames at the middle of first episode
    i = int(
        (dataset.meta.episodes[0]["dataset_to_index"] - dataset.meta.episodes[0]["dataset_from_index"]) / 2
    )
    load_and_compare(i)
    load_and_compare(i + 1)

    # test 2 last frames of first episode
    i = dataset.meta.episodes[0]["dataset_to_index"]
    load_and_compare(i - 2)
    load_and_compare(i - 1)


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


def test_check_cached_episodes_sufficient(tmp_path, lerobot_dataset_factory):
    """Test the _check_cached_episodes_sufficient method of LeRobotDataset."""
    # Create a dataset with 5 episodes (0-4)
    dataset = lerobot_dataset_factory(
        root=tmp_path / "test",
        total_episodes=5,
        total_frames=200,
        use_videos=False,
    )

    # Test hf_dataset is None
    dataset.hf_dataset = None
    assert dataset._check_cached_episodes_sufficient() is False

    # Test hf_dataset is empty
    import datasets

    empty_features = get_hf_features_from_features(dataset.features)
    dataset.hf_dataset = datasets.Dataset.from_dict(
        {key: [] for key in empty_features}, features=empty_features
    )
    dataset.hf_dataset.set_transform(hf_transform_to_torch)
    assert dataset._check_cached_episodes_sufficient() is False

    # Restore the original dataset for remaining tests
    dataset.hf_dataset = dataset.load_hf_dataset()

    # Test all episodes requested (self.episodes = None) and all are available
    dataset.episodes = None
    assert dataset._check_cached_episodes_sufficient() is True

    # Test specific episodes requested that are all available
    dataset.episodes = [0, 2, 4]
    assert dataset._check_cached_episodes_sufficient() is True

    # Test request episodes that don't exist in the cached dataset
    # Create a dataset with only episodes 0, 1, 2
    limited_dataset = lerobot_dataset_factory(
        root=tmp_path / "limited",
        total_episodes=3,
        total_frames=120,
        use_videos=False,
    )

    # Request episodes that include non-existent ones
    limited_dataset.episodes = [0, 1, 2, 3, 4]
    assert limited_dataset._check_cached_episodes_sufficient() is False

    # Test create a dataset with sparse episodes (e.g., only episodes 0, 2, 4)
    # First create the full dataset structure
    sparse_dataset = lerobot_dataset_factory(
        root=tmp_path / "sparse",
        total_episodes=5,
        total_frames=200,
        use_videos=False,
    )

    # Manually filter hf_dataset to only include episodes 0, 2, 4
    episode_indices = sparse_dataset.hf_dataset["episode_index"]
    mask = torch.zeros(len(episode_indices), dtype=torch.bool)
    for ep in [0, 2, 4]:
        mask |= torch.tensor(episode_indices) == ep

    # Create a filtered dataset
    filtered_data = {}
    # Find image keys by checking features
    image_keys = [key for key, ft in sparse_dataset.features.items() if ft.get("dtype") == "image"]

    for key in sparse_dataset.hf_dataset.column_names:
        values = sparse_dataset.hf_dataset[key]
        # Filter values based on mask
        filtered_values = [val for i, val in enumerate(values) if mask[i]]

        # Convert float32 image tensors back to uint8 numpy arrays for HuggingFace dataset
        if key in image_keys and len(filtered_values) > 0:
            # Convert torch tensors (float32, [0, 1], CHW) back to numpy arrays (uint8, [0, 255], HWC)
            filtered_values = [
                (val.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for val in filtered_values
            ]

        filtered_data[key] = filtered_values

    sparse_dataset.hf_dataset = datasets.Dataset.from_dict(
        filtered_data, features=get_hf_features_from_features(sparse_dataset.features)
    )
    sparse_dataset.hf_dataset.set_transform(hf_transform_to_torch)

    # Test requesting all episodes when only some are cached
    sparse_dataset.episodes = None
    assert sparse_dataset._check_cached_episodes_sufficient() is False

    # Test requesting only the available episodes
    sparse_dataset.episodes = [0, 2, 4]
    assert sparse_dataset._check_cached_episodes_sufficient() is True

    # Test requesting a mix of available and unavailable episodes
    sparse_dataset.episodes = [0, 1, 2]
    assert sparse_dataset._check_cached_episodes_sufficient() is False


def test_update_chunk_settings(tmp_path, empty_lerobot_dataset_factory):
    """Test the update_chunk_settings functionality for both LeRobotDataset and LeRobotDatasetMetadata."""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
        },
    }

    # Create dataset with default chunk settings
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)

    # Test initial default values
    initial_settings = dataset.meta.get_chunk_settings()
    assert initial_settings["chunks_size"] == DEFAULT_CHUNK_SIZE
    assert initial_settings["data_files_size_in_mb"] == DEFAULT_DATA_FILE_SIZE_IN_MB
    assert initial_settings["video_files_size_in_mb"] == DEFAULT_VIDEO_FILE_SIZE_IN_MB

    # Test updating all settings at once
    new_chunks_size = 2000
    new_data_size = 200
    new_video_size = 1000

    dataset.meta.update_chunk_settings(
        chunks_size=new_chunks_size,
        data_files_size_in_mb=new_data_size,
        video_files_size_in_mb=new_video_size,
    )

    # Verify settings were updated
    updated_settings = dataset.meta.get_chunk_settings()
    assert updated_settings["chunks_size"] == new_chunks_size
    assert updated_settings["data_files_size_in_mb"] == new_data_size
    assert updated_settings["video_files_size_in_mb"] == new_video_size

    # Test updating individual settings
    dataset.meta.update_chunk_settings(chunks_size=1500)
    settings_after_partial = dataset.meta.get_chunk_settings()
    assert settings_after_partial["chunks_size"] == 1500
    assert settings_after_partial["data_files_size_in_mb"] == new_data_size
    assert settings_after_partial["video_files_size_in_mb"] == new_video_size

    # Test updating only data file size
    dataset.meta.update_chunk_settings(data_files_size_in_mb=150)
    settings_after_data = dataset.meta.get_chunk_settings()
    assert settings_after_data["chunks_size"] == 1500
    assert settings_after_data["data_files_size_in_mb"] == 150
    assert settings_after_data["video_files_size_in_mb"] == new_video_size

    # Test updating only video file size
    dataset.meta.update_chunk_settings(video_files_size_in_mb=800)
    settings_after_video = dataset.meta.get_chunk_settings()
    assert settings_after_video["chunks_size"] == 1500
    assert settings_after_video["data_files_size_in_mb"] == 150
    assert settings_after_video["video_files_size_in_mb"] == 800

    # Test that settings persist in the info file
    info_path = dataset.root / "meta" / "info.json"
    assert info_path.exists()

    # Verify the underlying metadata properties
    assert dataset.meta.chunks_size == 1500
    assert dataset.meta.data_files_size_in_mb == 150
    assert dataset.meta.video_files_size_in_mb == 800

    # Test error handling for invalid values
    with pytest.raises(ValueError, match="chunks_size must be positive"):
        dataset.meta.update_chunk_settings(chunks_size=0)

    with pytest.raises(ValueError, match="chunks_size must be positive"):
        dataset.meta.update_chunk_settings(chunks_size=-100)

    with pytest.raises(ValueError, match="data_files_size_in_mb must be positive"):
        dataset.meta.update_chunk_settings(data_files_size_in_mb=0)

    with pytest.raises(ValueError, match="data_files_size_in_mb must be positive"):
        dataset.meta.update_chunk_settings(data_files_size_in_mb=-50)

    with pytest.raises(ValueError, match="video_files_size_in_mb must be positive"):
        dataset.meta.update_chunk_settings(video_files_size_in_mb=0)

    with pytest.raises(ValueError, match="video_files_size_in_mb must be positive"):
        dataset.meta.update_chunk_settings(video_files_size_in_mb=-200)

    # Test calling with None values (should not change anything)
    settings_before_none = dataset.meta.get_chunk_settings()
    dataset.meta.update_chunk_settings(
        chunks_size=None, data_files_size_in_mb=None, video_files_size_in_mb=None
    )
    settings_after_none = dataset.meta.get_chunk_settings()
    assert settings_before_none == settings_after_none

    # Test metadata direct access
    meta_settings = dataset.meta.get_chunk_settings()
    assert meta_settings == dataset.meta.get_chunk_settings()

    # Test updating via metadata directly
    dataset.meta.update_chunk_settings(chunks_size=3000)
    assert dataset.meta.get_chunk_settings()["chunks_size"] == 3000


def test_update_chunk_settings_video_dataset(tmp_path):
    """Test update_chunk_settings with a video dataset to ensure video-specific logic works."""
    features = {
        "observation.images.cam": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {"dtype": "float32", "shape": (6,), "names": ["j1", "j2", "j3", "j4", "j5", "j6"]},
    }

    # Create video dataset
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=30, features=features, root=tmp_path / "video_test", use_videos=True
    )

    # Test that video-specific settings work
    original_video_size = dataset.meta.get_chunk_settings()["video_files_size_in_mb"]
    new_video_size = original_video_size * 2

    dataset.meta.update_chunk_settings(video_files_size_in_mb=new_video_size)
    assert dataset.meta.get_chunk_settings()["video_files_size_in_mb"] == new_video_size
    assert dataset.meta.video_files_size_in_mb == new_video_size
