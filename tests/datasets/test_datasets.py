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
    VALID_VIDEO_CODECS,
    LeRobotDataset,
    MultiLeRobotDataset,
    _encode_video_worker,
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
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, OBS_STR, REWARD
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
    action_features = hw_to_dataset_features(robot.action_features, ACTION, True)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR, True)
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
    from lerobot.utils.constants import HF_LEROBOT_HOME

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


def test_tmp_image_deletion(tmp_path, empty_lerobot_dataset_factory):
    """Verify temporary image directories are removed for image features after saving episode."""
    # Image feature: images should be deleted after saving episode
    image_key = "image"
    features_image = {
        image_key: {"dtype": "image", "shape": DUMMY_CHW, "names": ["channels", "height", "width"]}
    }
    ds_img = empty_lerobot_dataset_factory(root=tmp_path / "img", features=features_image)
    ds_img.add_frame({"image": np.random.rand(*DUMMY_CHW), "task": "Dummy task"})
    ds_img.save_episode()
    img_dir = ds_img._get_image_file_dir(0, image_key)
    assert not img_dir.exists(), "Temporary image directory should be removed for image features"


def test_tmp_video_deletion(tmp_path, empty_lerobot_dataset_factory):
    """Verify temporary image directories are removed for video encoding when `batch_encoding_size == 1`."""
    # Video feature: when batch_encoding_size == 1 temporary images should be deleted
    vid_key = "video"
    features_video = {
        vid_key: {"dtype": "video", "shape": DUMMY_CHW, "names": ["channels", "height", "width"]}
    }

    ds_vid = empty_lerobot_dataset_factory(root=tmp_path / "vid", features=features_video)
    ds_vid.batch_encoding_size = 1
    ds_vid.add_frame({vid_key: np.random.rand(*DUMMY_CHW), "task": "Dummy task"})
    ds_vid.save_episode()
    vid_img_dir = ds_vid._get_image_file_dir(0, vid_key)
    assert not vid_img_dir.exists(), (
        "Temporary image directory should be removed when batch_encoding_size == 1"
    )


def test_tmp_mixed_deletion(tmp_path, empty_lerobot_dataset_factory):
    """Verify temporary image directories are removed appropriately when both image and video features are present."""
    image_key = "image"
    vid_key = "video"
    features_mixed = {
        image_key: {"dtype": "image", "shape": DUMMY_CHW, "names": ["channels", "height", "width"]},
        vid_key: {"dtype": "video", "shape": DUMMY_HWC, "names": ["height", "width", "channels"]},
    }
    ds_mixed = empty_lerobot_dataset_factory(
        root=tmp_path / "mixed", features=features_mixed, batch_encoding_size=2
    )
    ds_mixed.add_frame(
        {
            "image": np.random.rand(*DUMMY_CHW),
            "video": np.random.rand(*DUMMY_HWC),
            "task": "Dummy task",
        }
    )
    ds_mixed.save_episode()
    img_dir = ds_mixed._get_image_file_dir(0, image_key)
    vid_img_dir = ds_mixed._get_image_file_dir(0, vid_key)
    assert not img_dir.exists(), "Temporary image directory should be removed for image features"
    assert vid_img_dir.exists(), (
        "Temporary image directory should not be removed for video features when batch_encoding_size == 2"
    )


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
        (ACTION, 1, True),
        ("episode_index", 0, True),
        ("frame_index", 0, True),
        ("timestamp", 0, True),
        # TODO(rcadene): should we rename it agent_pos?
        (OBS_STATE, 1, True),
        (REWARD, 0, False),
        (DONE, 0, False),
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
        OBS_STATE: {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
        },
        ACTION: {
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
        f"{OBS_IMAGES}.cam": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        ACTION: {"dtype": "float32", "shape": (6,), "names": ["j1", "j2", "j3", "j4", "j5", "j6"]},
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


def test_episode_index_distribution(tmp_path, empty_lerobot_dataset_factory):
    """Test that all frames have correct episode indices across multiple episodes."""
    features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, use_videos=False)

    # Create 3 episodes with different lengths
    num_episodes = 3
    frames_per_episode = [10, 15, 8]

    for episode_idx in range(num_episodes):
        for _ in range(frames_per_episode[episode_idx]):
            dataset.add_frame({"state": torch.randn(2), "task": f"task_{episode_idx}"})
        dataset.save_episode()

    dataset.finalize()

    # Load the dataset and check episode indices
    loaded_dataset = LeRobotDataset(dataset.repo_id, root=dataset.root)

    # Check specific frames across episode boundaries
    cumulative = 0
    for ep_idx, ep_length in enumerate(frames_per_episode):
        # Check start, middle, and end of each episode
        start_frame = cumulative
        middle_frame = cumulative + ep_length // 2
        end_frame = cumulative + ep_length - 1

        for frame_idx in [start_frame, middle_frame, end_frame]:
            frame_data = loaded_dataset[frame_idx]
            actual_ep_idx = frame_data["episode_index"].item()
            assert actual_ep_idx == ep_idx, (
                f"Frame {frame_idx} has episode_index {actual_ep_idx}, should be {ep_idx}"
            )

        cumulative += ep_length

    # Check episode index distribution
    all_episode_indices = [loaded_dataset[i]["episode_index"].item() for i in range(len(loaded_dataset))]
    from collections import Counter

    distribution = Counter(all_episode_indices)
    expected_dist = {i: frames_per_episode[i] for i in range(num_episodes)}

    assert dict(distribution) == expected_dist, (
        f"Episode distribution {dict(distribution)} != expected {expected_dist}"
    )


def test_multi_episode_metadata_consistency(tmp_path, empty_lerobot_dataset_factory):
    """Test episode metadata consistency across multiple episodes."""
    features = {
        "state": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        ACTION: {"dtype": "float32", "shape": (2,), "names": ["v", "w"]},
    }
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, use_videos=False)

    num_episodes = 4
    frames_per_episode = [20, 35, 10, 25]
    tasks = ["pick", "place", "pick", "place"]

    for episode_idx in range(num_episodes):
        for _ in range(frames_per_episode[episode_idx]):
            dataset.add_frame({"state": torch.randn(3), ACTION: torch.randn(2), "task": tasks[episode_idx]})
        dataset.save_episode()

    dataset.finalize()

    # Load and validate episode metadata
    loaded_dataset = LeRobotDataset(dataset.repo_id, root=dataset.root)

    assert loaded_dataset.meta.total_episodes == num_episodes
    assert loaded_dataset.meta.total_frames == sum(frames_per_episode)

    cumulative_frames = 0
    for episode_idx in range(num_episodes):
        episode_metadata = loaded_dataset.meta.episodes[episode_idx]

        # Check basic episode properties
        assert episode_metadata["episode_index"] == episode_idx
        assert episode_metadata["length"] == frames_per_episode[episode_idx]
        assert episode_metadata["tasks"] == [tasks[episode_idx]]

        # Check dataset indices
        expected_from = cumulative_frames
        expected_to = cumulative_frames + frames_per_episode[episode_idx]

        assert episode_metadata["dataset_from_index"] == expected_from
        assert episode_metadata["dataset_to_index"] == expected_to

        cumulative_frames += frames_per_episode[episode_idx]


def test_data_consistency_across_episodes(tmp_path, empty_lerobot_dataset_factory):
    """Test that episodes have no gaps or overlaps in their data indices."""
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, use_videos=False)

    num_episodes = 5
    frames_per_episode = [12, 8, 20, 15, 5]

    for episode_idx in range(num_episodes):
        for _ in range(frames_per_episode[episode_idx]):
            dataset.add_frame({"state": torch.randn(1), "task": "consistency_test"})
        dataset.save_episode()

    dataset.finalize()

    loaded_dataset = LeRobotDataset(dataset.repo_id, root=dataset.root)

    # Check data consistency - no gaps or overlaps
    cumulative_check = 0
    for episode_idx in range(num_episodes):
        episode_metadata = loaded_dataset.meta.episodes[episode_idx]
        from_idx = episode_metadata["dataset_from_index"]
        to_idx = episode_metadata["dataset_to_index"]

        # Check that episode starts exactly where previous ended
        assert from_idx == cumulative_check, (
            f"Episode {episode_idx} starts at {from_idx}, expected {cumulative_check}"
        )

        # Check that episode length matches expected
        actual_length = to_idx - from_idx
        expected_length = frames_per_episode[episode_idx]
        assert actual_length == expected_length, (
            f"Episode {episode_idx} length {actual_length} != expected {expected_length}"
        )

        cumulative_check = to_idx

    # Final check: last episode should end at total frames
    expected_total_frames = sum(frames_per_episode)
    assert cumulative_check == expected_total_frames, (
        f"Final frame count {cumulative_check} != expected {expected_total_frames}"
    )


def test_statistics_metadata_validation(tmp_path, empty_lerobot_dataset_factory):
    """Test that statistics are properly computed and stored for all features."""
    features = {
        "state": {"dtype": "float32", "shape": (2,), "names": ["pos", "vel"]},
        ACTION: {"dtype": "float32", "shape": (1,), "names": ["force"]},
    }
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, use_videos=False)

    # Create controlled data to verify statistics
    num_episodes = 2
    frames_per_episode = [10, 10]

    # Use deterministic data for predictable statistics
    torch.manual_seed(42)
    for episode_idx in range(num_episodes):
        for frame_idx in range(frames_per_episode[episode_idx]):
            state_data = torch.tensor([frame_idx * 0.1, frame_idx * 0.2], dtype=torch.float32)
            action_data = torch.tensor([frame_idx * 0.05], dtype=torch.float32)
            dataset.add_frame({"state": state_data, ACTION: action_data, "task": "stats_test"})
        dataset.save_episode()

    dataset.finalize()

    loaded_dataset = LeRobotDataset(dataset.repo_id, root=dataset.root)

    # Check that statistics exist for all features
    assert loaded_dataset.meta.stats is not None, "No statistics found"

    for feature_name in features:
        assert feature_name in loaded_dataset.meta.stats, f"No statistics for feature '{feature_name}'"

        feature_stats = loaded_dataset.meta.stats[feature_name]
        expected_stats = ["min", "max", "mean", "std", "count"]

        for stat_key in expected_stats:
            assert stat_key in feature_stats, f"Missing '{stat_key}' statistic for '{feature_name}'"

            stat_value = feature_stats[stat_key]
            # Basic sanity checks
            if stat_key == "count":
                assert stat_value == sum(frames_per_episode), f"Wrong count for '{feature_name}'"
            elif stat_key in ["min", "max", "mean", "std"]:
                # Check that statistics are reasonable (not NaN, proper shapes)
                if hasattr(stat_value, "shape"):
                    expected_shape = features[feature_name]["shape"]
                    assert stat_value.shape == expected_shape or len(stat_value) == expected_shape[0], (
                        f"Wrong shape for {stat_key} of '{feature_name}'"
                    )
                # Check no NaN values
                if hasattr(stat_value, "__iter__"):
                    assert not any(np.isnan(v) for v in stat_value), f"NaN in {stat_key} for '{feature_name}'"
                else:
                    assert not np.isnan(stat_value), f"NaN in {stat_key} for '{feature_name}'"


def test_episode_boundary_integrity(tmp_path, empty_lerobot_dataset_factory):
    """Test frame indices and episode transitions at episode boundaries."""
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, use_videos=False)

    num_episodes = 3
    frames_per_episode = [7, 12, 5]

    for episode_idx in range(num_episodes):
        for frame_idx in range(frames_per_episode[episode_idx]):
            dataset.add_frame({"state": torch.tensor([float(frame_idx)]), "task": f"episode_{episode_idx}"})
        dataset.save_episode()

    dataset.finalize()

    loaded_dataset = LeRobotDataset(dataset.repo_id, root=dataset.root)

    # Test episode boundaries
    cumulative = 0
    for ep_idx, ep_length in enumerate(frames_per_episode):
        if ep_idx > 0:
            # Check last frame of previous episode
            prev_frame = loaded_dataset[cumulative - 1]
            assert prev_frame["episode_index"].item() == ep_idx - 1

        # Check first frame of current episode
        if cumulative < len(loaded_dataset):
            curr_frame = loaded_dataset[cumulative]
            assert curr_frame["episode_index"].item() == ep_idx

        # Check frame_index within episode
        for i in range(ep_length):
            if cumulative + i < len(loaded_dataset):
                frame = loaded_dataset[cumulative + i]
                assert frame["frame_index"].item() == i, f"Frame {cumulative + i} has wrong frame_index"
                assert frame["episode_index"].item() == ep_idx, (
                    f"Frame {cumulative + i} has wrong episode_index"
                )

        cumulative += ep_length


def test_task_indexing_and_validation(tmp_path, empty_lerobot_dataset_factory):
    """Test that tasks are properly indexed and retrievable."""
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, use_videos=False)

    # Use multiple tasks, including repeated ones
    tasks = ["pick", "place", "pick", "navigate", "place"]
    unique_tasks = list(set(tasks))  # ["pick", "place", "navigate"]
    frames_per_episode = [5, 8, 3, 10, 6]

    for episode_idx, task in enumerate(tasks):
        for _ in range(frames_per_episode[episode_idx]):
            dataset.add_frame({"state": torch.randn(1), "task": task})
        dataset.save_episode()

    dataset.finalize()

    loaded_dataset = LeRobotDataset(dataset.repo_id, root=dataset.root)

    # Check that all unique tasks are in the tasks metadata
    stored_tasks = set(loaded_dataset.meta.tasks.index)
    assert stored_tasks == set(unique_tasks), f"Stored tasks {stored_tasks} != expected {set(unique_tasks)}"

    # Check that task indices are consistent
    cumulative = 0
    for episode_idx, expected_task in enumerate(tasks):
        episode_metadata = loaded_dataset.meta.episodes[episode_idx]
        assert episode_metadata["tasks"] == [expected_task]

        # Check frames in this episode have correct task
        for i in range(frames_per_episode[episode_idx]):
            frame = loaded_dataset[cumulative + i]
            assert frame["task"] == expected_task, f"Frame {cumulative + i} has wrong task"

            # Check task_index consistency
            expected_task_index = loaded_dataset.meta.get_task_index(expected_task)
            assert frame["task_index"].item() == expected_task_index

        cumulative += frames_per_episode[episode_idx]

    # Check total number of tasks
    assert loaded_dataset.meta.total_tasks == len(unique_tasks)


def test_dataset_resume_recording(tmp_path, empty_lerobot_dataset_factory):
    """Test that resuming dataset recording preserves previously recorded episodes.

    This test validates the critical resume functionality by:
    1. Recording initial episodes and finalizing
    2. Reopening the dataset
    3. Recording additional episodes
    4. Verifying all data (old + new) is intact

    This specifically tests the bug fix where parquet files were being overwritten
    instead of appended to during resume.
    """
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
    }

    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, use_videos=False)

    initial_episodes = 2
    frames_per_episode = 3

    for ep_idx in range(initial_episodes):
        for frame_idx in range(frames_per_episode):
            dataset.add_frame(
                {
                    "observation.state": torch.tensor([float(ep_idx), float(frame_idx)]),
                    "action": torch.tensor([0.5, 0.5]),
                    "task": f"task_{ep_idx}",
                }
            )
        dataset.save_episode()

    assert dataset.meta.total_episodes == initial_episodes
    assert dataset.meta.total_frames == initial_episodes * frames_per_episode

    dataset.finalize()
    initial_root = dataset.root
    initial_repo_id = dataset.repo_id
    del dataset

    dataset_verify = LeRobotDataset(initial_repo_id, root=initial_root, revision="v3.0")
    assert dataset_verify.meta.total_episodes == initial_episodes
    assert dataset_verify.meta.total_frames == initial_episodes * frames_per_episode
    assert len(dataset_verify.hf_dataset) == initial_episodes * frames_per_episode

    for idx in range(len(dataset_verify.hf_dataset)):
        item = dataset_verify[idx]
        expected_ep = idx // frames_per_episode
        expected_frame = idx % frames_per_episode
        assert item["episode_index"].item() == expected_ep
        assert item["frame_index"].item() == expected_frame
        assert item["index"].item() == idx
        assert item["observation.state"][0].item() == float(expected_ep)
        assert item["observation.state"][1].item() == float(expected_frame)

    del dataset_verify

    # Phase 3: Resume recording - add more episodes
    dataset_resumed = LeRobotDataset(initial_repo_id, root=initial_root, revision="v3.0")

    assert dataset_resumed.meta.total_episodes == initial_episodes
    assert dataset_resumed.meta.total_frames == initial_episodes * frames_per_episode
    assert dataset_resumed.latest_episode is None  # Not recording yet
    assert dataset_resumed.writer is None
    assert dataset_resumed.meta.writer is None

    additional_episodes = 2
    for ep_idx in range(initial_episodes, initial_episodes + additional_episodes):
        for frame_idx in range(frames_per_episode):
            dataset_resumed.add_frame(
                {
                    "observation.state": torch.tensor([float(ep_idx), float(frame_idx)]),
                    "action": torch.tensor([0.5, 0.5]),
                    "task": f"task_{ep_idx}",
                }
            )
        dataset_resumed.save_episode()

    total_episodes = initial_episodes + additional_episodes
    total_frames = total_episodes * frames_per_episode
    assert dataset_resumed.meta.total_episodes == total_episodes
    assert dataset_resumed.meta.total_frames == total_frames

    dataset_resumed.finalize()
    del dataset_resumed

    dataset_final = LeRobotDataset(initial_repo_id, root=initial_root, revision="v3.0")

    assert dataset_final.meta.total_episodes == total_episodes
    assert dataset_final.meta.total_frames == total_frames
    assert len(dataset_final.hf_dataset) == total_frames

    for idx in range(total_frames):
        item = dataset_final[idx]
        expected_ep = idx // frames_per_episode
        expected_frame = idx % frames_per_episode

        assert item["episode_index"].item() == expected_ep, (
            f"Frame {idx}: wrong episode_index. Expected {expected_ep}, got {item['episode_index'].item()}"
        )
        assert item["frame_index"].item() == expected_frame, (
            f"Frame {idx}: wrong frame_index. Expected {expected_frame}, got {item['frame_index'].item()}"
        )
        assert item["index"].item() == idx, (
            f"Frame {idx}: wrong index. Expected {idx}, got {item['index'].item()}"
        )

        # Verify data integrity
        assert item["observation.state"][0].item() == float(expected_ep), (
            f"Frame {idx}: wrong observation.state[0]. Expected {float(expected_ep)}, "
            f"got {item['observation.state'][0].item()}"
        )
        assert item["observation.state"][1].item() == float(expected_frame), (
            f"Frame {idx}: wrong observation.state[1]. Expected {float(expected_frame)}, "
            f"got {item['observation.state'][1].item()}"
        )

    assert len(dataset_final.meta.episodes) == total_episodes
    for ep_idx in range(total_episodes):
        ep_metadata = dataset_final.meta.episodes[ep_idx]
        assert ep_metadata["episode_index"] == ep_idx
        assert ep_metadata["length"] == frames_per_episode
        assert ep_metadata["tasks"] == [f"task_{ep_idx}"]

        expected_from = ep_idx * frames_per_episode
        expected_to = (ep_idx + 1) * frames_per_episode
        assert ep_metadata["dataset_from_index"] == expected_from
        assert ep_metadata["dataset_to_index"] == expected_to


def test_frames_in_current_file_calculation(tmp_path, empty_lerobot_dataset_factory):
    """Regression test for bug where frames_in_current_file only counted frames from last episode instead of all frames in current file."""
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["vx", "vy"]},
    }

    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, use_videos=False)
    dataset.meta.update_chunk_settings(data_files_size_in_mb=100)

    assert dataset._current_file_start_frame is None

    frames_per_episode = 10
    for _ in range(frames_per_episode):
        dataset.add_frame(
            {
                "observation.state": torch.randn(2),
                "action": torch.randn(2),
                "task": "task_0",
            }
        )
    dataset.save_episode()

    assert dataset._current_file_start_frame == 0
    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == frames_per_episode

    for _ in range(frames_per_episode):
        dataset.add_frame(
            {
                "observation.state": torch.randn(2),
                "action": torch.randn(2),
                "task": "task_1",
            }
        )
    dataset.save_episode()

    assert dataset._current_file_start_frame == 0
    assert dataset.meta.total_episodes == 2
    assert dataset.meta.total_frames == 2 * frames_per_episode

    ep1_chunk = dataset.latest_episode["data/chunk_index"]
    ep1_file = dataset.latest_episode["data/file_index"]
    assert ep1_chunk == 0
    assert ep1_file == 0

    for _ in range(frames_per_episode):
        dataset.add_frame(
            {
                "observation.state": torch.randn(2),
                "action": torch.randn(2),
                "task": "task_2",
            }
        )
    dataset.save_episode()

    assert dataset._current_file_start_frame == 0
    assert dataset.meta.total_episodes == 3
    assert dataset.meta.total_frames == 3 * frames_per_episode

    ep2_chunk = dataset.latest_episode["data/chunk_index"]
    ep2_file = dataset.latest_episode["data/file_index"]
    assert ep2_chunk == 0
    assert ep2_file == 0

    dataset.finalize()

    from lerobot.datasets.utils import load_episodes

    dataset.meta.episodes = load_episodes(dataset.root)
    assert dataset.meta.episodes is not None

    for ep_idx in range(3):
        ep_metadata = dataset.meta.episodes[ep_idx]
        assert ep_metadata["data/chunk_index"] == 0
        assert ep_metadata["data/file_index"] == 0

        expected_from = ep_idx * frames_per_episode
        expected_to = (ep_idx + 1) * frames_per_episode
        assert ep_metadata["dataset_from_index"] == expected_from
        assert ep_metadata["dataset_to_index"] == expected_to

    loaded_dataset = LeRobotDataset(dataset.repo_id, root=dataset.root)
    assert len(loaded_dataset) == 3 * frames_per_episode
    assert loaded_dataset.meta.total_episodes == 3
    assert loaded_dataset.meta.total_frames == 3 * frames_per_episode

    for idx in range(len(loaded_dataset)):
        frame = loaded_dataset[idx]
        expected_ep = idx // frames_per_episode
        assert frame["episode_index"].item() == expected_ep


def test_encode_video_worker_forwards_vcodec(tmp_path):
    """Test that _encode_video_worker correctly forwards the vcodec parameter to encode_video_frames."""
    from unittest.mock import patch

    from lerobot.datasets.utils import DEFAULT_IMAGE_PATH

    # Create the expected directory structure
    video_key = "observation.images.laptop"
    episode_index = 0
    frame_index = 0

    fpath = DEFAULT_IMAGE_PATH.format(
        image_key=video_key, episode_index=episode_index, frame_index=frame_index
    )
    img_dir = tmp_path / Path(fpath).parent
    img_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy image file
    dummy_img = Image.new("RGB", (64, 64), color="red")
    dummy_img.save(img_dir / "frame-000000.png")

    # Track what vcodec was passed to encode_video_frames
    captured_kwargs = {}

    def mock_encode_video_frames(imgs_dir, video_path, fps, **kwargs):
        captured_kwargs.update(kwargs)
        # Create a dummy output file so the worker doesn't fail
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).touch()

    with patch("lerobot.datasets.lerobot_dataset.encode_video_frames", side_effect=mock_encode_video_frames):
        # Test with h264 codec
        _encode_video_worker(video_key, episode_index, tmp_path, fps=30, vcodec="h264")

    assert "vcodec" in captured_kwargs
    assert captured_kwargs["vcodec"] == "h264"


def test_encode_video_worker_default_vcodec(tmp_path):
    """Test that _encode_video_worker uses libsvtav1 as the default codec."""
    from unittest.mock import patch

    from lerobot.datasets.utils import DEFAULT_IMAGE_PATH

    # Create the expected directory structure
    video_key = "observation.images.laptop"
    episode_index = 0
    frame_index = 0

    fpath = DEFAULT_IMAGE_PATH.format(
        image_key=video_key, episode_index=episode_index, frame_index=frame_index
    )
    img_dir = tmp_path / Path(fpath).parent
    img_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy image file
    dummy_img = Image.new("RGB", (64, 64), color="red")
    dummy_img.save(img_dir / "frame-000000.png")

    # Track what vcodec was passed to encode_video_frames
    captured_kwargs = {}

    def mock_encode_video_frames(imgs_dir, video_path, fps, **kwargs):
        captured_kwargs.update(kwargs)
        # Create a dummy output file so the worker doesn't fail
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).touch()

    with patch("lerobot.datasets.lerobot_dataset.encode_video_frames", side_effect=mock_encode_video_frames):
        # Test with default codec (no vcodec specified)
        _encode_video_worker(video_key, episode_index, tmp_path, fps=30)

    assert "vcodec" in captured_kwargs
    assert captured_kwargs["vcodec"] == "libsvtav1"


def test_lerobot_dataset_vcodec_validation():
    """Test that LeRobotDataset validates the vcodec parameter."""
    # Test that invalid vcodec raises ValueError
    with pytest.raises(ValueError, match="Invalid vcodec"):
        LeRobotDataset.__new__(LeRobotDataset)  # bypass __init__ to test validation directly
        # Actually test via create since it's easier
        LeRobotDataset.create(
            repo_id="test/invalid_codec",
            fps=30,
            features={"observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]}},
            vcodec="invalid_codec",
        )


def test_valid_video_codecs_constant():
    """Test that VALID_VIDEO_CODECS contains the expected codecs."""
    assert "h264" in VALID_VIDEO_CODECS
    assert "hevc" in VALID_VIDEO_CODECS
    assert "libsvtav1" in VALID_VIDEO_CODECS
    assert len(VALID_VIDEO_CODECS) == 3


def test_delta_timestamps_with_episodes_filter(tmp_path, empty_lerobot_dataset_factory):
    """Regression test for bug where delta_timestamps incorrectly marked all frames as padded when using episodes filter.

    The bug occurred because _get_query_indices was using the relative index (idx) in the filtered dataset
    instead of the absolute index when comparing against episode boundaries (ep_start, ep_end).
    """
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["vx", "vy"]},
    }

    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, use_videos=False)

    # Create 3 episodes with 10 frames each
    frames_per_episode = 10
    for ep_idx in range(3):
        for frame_idx in range(frames_per_episode):
            dataset.add_frame(
                {
                    "observation.state": torch.tensor([ep_idx, frame_idx], dtype=torch.float32),
                    "action": torch.randn(2),
                    "task": f"task_{ep_idx}",
                }
            )
        dataset.save_episode()
    dataset.finalize()

    # Load only episode 1 (middle episode) with delta_timestamps
    delta_ts = {"observation.state": [0.0]}  # Just the current frame
    filtered_dataset = LeRobotDataset(
        dataset.repo_id,
        root=dataset.root,
        episodes=[1],
        delta_timestamps=delta_ts,
    )

    # Verify the filtered dataset has the correct length
    assert len(filtered_dataset) == frames_per_episode

    # Check that no frames are marked as padded (since delta=0 should always be valid)
    for idx in range(len(filtered_dataset)):
        frame = filtered_dataset[idx]
        assert frame["observation.state_is_pad"].item() is False, f"Frame {idx} incorrectly marked as padded"
        # Verify we're getting data from episode 1
        assert frame["episode_index"].item() == 1


def test_delta_timestamps_padding_at_episode_boundaries(tmp_path, empty_lerobot_dataset_factory):
    """Test that delta_timestamps correctly marks padding at episode boundaries when using episodes filter."""
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["vx", "vy"]},
    }

    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "test", features=features, use_videos=False, fps=10
    )

    # Create 3 episodes with 5 frames each
    frames_per_episode = 5
    for ep_idx in range(3):
        for frame_idx in range(frames_per_episode):
            dataset.add_frame(
                {
                    "observation.state": torch.tensor([ep_idx, frame_idx], dtype=torch.float32),
                    "action": torch.randn(2),
                    "task": f"task_{ep_idx}",
                }
            )
        dataset.save_episode()
    dataset.finalize()

    # Load only episode 1 with delta_timestamps that go beyond episode boundaries
    # fps=10, so 0.1s = 1 frame offset
    delta_ts = {"observation.state": [-0.2, -0.1, 0.0, 0.1, 0.2]}  # -2, -1, 0, +1, +2 frames
    filtered_dataset = LeRobotDataset(
        dataset.repo_id,
        root=dataset.root,
        episodes=[1],
        delta_timestamps=delta_ts,
        tolerance_s=0.04,  # Slightly less than half a frame at 10fps
    )

    assert len(filtered_dataset) == frames_per_episode

    # Check padding at the start of the episode (first frame)
    first_frame = filtered_dataset[0]
    is_pad = first_frame["observation.state_is_pad"].tolist()
    # At frame 0 of episode 1: delta -2 and -1 should be padded, 0, +1, +2 should not
    assert is_pad == [True, True, False, False, False], f"First frame padding incorrect: {is_pad}"

    # Check middle frame (no padding expected)
    mid_frame = filtered_dataset[2]
    is_pad = mid_frame["observation.state_is_pad"].tolist()
    assert is_pad == [False, False, False, False, False], f"Middle frame padding incorrect: {is_pad}"

    # Check padding at the end of the episode (last frame)
    last_frame = filtered_dataset[4]
    is_pad = last_frame["observation.state_is_pad"].tolist()
    # At frame 4 of episode 1: delta -2, -1, 0 should not be padded, +1, +2 should be
    assert is_pad == [False, False, False, True, True], f"Last frame padding incorrect: {is_pad}"


def test_delta_timestamps_multiple_episodes_filter(tmp_path, empty_lerobot_dataset_factory):
    """Test delta_timestamps with multiple non-consecutive episodes selected."""
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
    }

    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "test", features=features, use_videos=False, fps=10
    )

    # Create 5 episodes with 5 frames each
    frames_per_episode = 5
    for ep_idx in range(5):
        for frame_idx in range(frames_per_episode):
            dataset.add_frame(
                {
                    "observation.state": torch.tensor([ep_idx, frame_idx], dtype=torch.float32),
                    "task": f"task_{ep_idx}",
                }
            )
        dataset.save_episode()
    dataset.finalize()

    # Load episodes 1 and 3 (non-consecutive)
    delta_ts = {"observation.state": [0.0]}
    filtered_dataset = LeRobotDataset(
        dataset.repo_id,
        root=dataset.root,
        episodes=[1, 3],
        delta_timestamps=delta_ts,
    )

    assert len(filtered_dataset) == 2 * frames_per_episode

    # All frames should have valid (non-padded) data for delta=0
    for idx in range(len(filtered_dataset)):
        frame = filtered_dataset[idx]
        assert frame["observation.state_is_pad"].item() is False

    # Verify we're getting the correct episodes
    episode_indices = [filtered_dataset[i]["episode_index"].item() for i in range(len(filtered_dataset))]
    expected_episodes = [1] * frames_per_episode + [3] * frames_per_episode
    assert episode_indices == expected_episodes


def test_delta_timestamps_query_returns_correct_values(tmp_path, empty_lerobot_dataset_factory):
    """Test that delta_timestamps returns the correct observation values, not just correct padding."""
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["x"]},
    }

    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "test", features=features, use_videos=False, fps=10
    )

    # Create 2 episodes with known values
    # Episode 0: frames with values 0, 1, 2, 3, 4
    # Episode 1: frames with values 10, 11, 12, 13, 14
    frames_per_episode = 5
    for ep_idx in range(2):
        for frame_idx in range(frames_per_episode):
            value = ep_idx * 10 + frame_idx
            dataset.add_frame(
                {
                    "observation.state": torch.tensor([value], dtype=torch.float32),
                    "task": f"task_{ep_idx}",
                }
            )
        dataset.save_episode()
    dataset.finalize()

    # Load episode 1 with delta that looks at previous frame
    delta_ts = {"observation.state": [-0.1, 0.0]}  # Previous frame and current frame
    filtered_dataset = LeRobotDataset(
        dataset.repo_id,
        root=dataset.root,
        episodes=[1],
        delta_timestamps=delta_ts,
        tolerance_s=0.04,
    )

    # Check frame 2 of episode 1 (which has absolute index 7, value 12)
    frame = filtered_dataset[2]
    state_values = frame["observation.state"].tolist()
    # Should get [11, 12] - the previous and current values within episode 1
    assert state_values == [11.0, 12.0], f"Expected [11.0, 12.0], got {state_values}"

    # Check first frame - previous frame should be clamped to episode start (padded)
    first_frame = filtered_dataset[0]
    state_values = first_frame["observation.state"].tolist()
    is_pad = first_frame["observation.state_is_pad"].tolist()
    # Previous frame is outside episode, so it's clamped to first frame and marked as padded
    assert state_values == [10.0, 10.0], f"Expected [10.0, 10.0], got {state_values}"
    assert is_pad == [True, False], f"Expected [True, False], got {is_pad}"
