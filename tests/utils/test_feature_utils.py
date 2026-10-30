#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

ND_FEATURE_SHAPE = (16, 3)
CAMERA_SHAPE = (8, 8, 3)


def _hw_observation_features() -> dict:
    """A robot-style declaration mixing a scalar, a camera and an N-D numeric array."""
    return {
        "joint.pos": float,
        "cam": CAMERA_SHAPE,
        "sensor": PolicyFeature(type=FeatureType.STATE, shape=ND_FEATURE_SHAPE),
    }


def _nd_dataset_features() -> dict[str, dict]:
    return {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
        "observation.sensor": {"dtype": "float32", "shape": ND_FEATURE_SHAPE, "names": None},
    }


# ── hw_to_dataset_features ────────────────────────────────────────────


def test_hw_to_dataset_features_nd_numeric_feature():
    """A fixed-shape numeric array that is not an image becomes a standalone float32 feature."""
    features = hw_to_dataset_features(_hw_observation_features(), OBS_STR, use_video=True)

    assert features["observation.sensor"]["dtype"] == "float32"
    assert features["observation.sensor"]["shape"] == ND_FEATURE_SHAPE
    assert features["observation.sensor"]["names"] is None
    assert features["observation.state"]["names"] == ["joint.pos"]
    assert features["observation.images.cam"]["dtype"] == "video"


def test_hw_to_dataset_features_nd_numeric_feature_action_prefix():
    hw_features = {"joint.pos": float, "chunk": PolicyFeature(type=FeatureType.ACTION, shape=(4, 2))}

    features = hw_to_dataset_features(hw_features, ACTION)

    assert features[ACTION] == {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]}
    assert features["action.chunk"] == {"dtype": "float32", "shape": (4, 2), "names": None}


def test_hw_to_dataset_features_3d_numeric_feature_is_not_an_image():
    hw_features = {"sensor": PolicyFeature(type=FeatureType.STATE, shape=(4, 16, 3))}

    features = hw_to_dataset_features(hw_features, OBS_STR)

    assert features == {"observation.sensor": {"dtype": "float32", "shape": (4, 16, 3), "names": None}}


def test_hw_to_dataset_features_scalars_unchanged():
    """``float`` entries and 1-D ``PolicyFeature`` entries still fold into the state/action vector."""
    hw_features = {"a.pos": float, "ee.x": PolicyFeature(type=FeatureType.STATE, shape=(1,))}

    assert hw_to_dataset_features(hw_features, OBS_STR) == {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a.pos", "ee.x"]}
    }
    assert hw_to_dataset_features({"a.pos": float}, ACTION) == {
        ACTION: {"dtype": "float32", "shape": (1,), "names": ["a.pos"]}
    }


@pytest.mark.parametrize("use_video", [True, False])
@pytest.mark.parametrize("channels,is_depth", [(3, False), (1, True)])
def test_hw_to_dataset_features_cameras_unchanged(use_video, channels, is_depth):
    shape = (480, 640, channels)

    features = hw_to_dataset_features({"cam": shape}, OBS_STR, use_video=use_video)

    assert features == {
        "observation.images.cam": {
            "dtype": "video" if use_video else "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": is_depth},
        }
    }


def test_hw_to_dataset_features_invalid_camera_shape_still_raises():
    with pytest.raises(ValueError, match="Expected a 3-tuple"):
        hw_to_dataset_features({"cam": (480, 640, 2)}, OBS_STR)


# ── build_dataset_frame ───────────────────────────────────────────────


@pytest.mark.parametrize("input_dtype", [np.float32, np.float64])
def test_build_dataset_frame_passes_nd_numeric_array_through(input_dtype):
    """``build_dataset_frame`` copies an N-D array into the frame under its dataset key, as float32."""
    sensor = np.arange(np.prod(ND_FEATURE_SHAPE), dtype=input_dtype).reshape(ND_FEATURE_SHAPE)
    values = {"joint.pos": 1.0, "sensor": sensor}

    frame = build_dataset_frame(_nd_dataset_features(), values, OBS_STR)

    assert "observation.sensor" in frame
    assert frame["observation.sensor"].dtype == np.float32
    assert frame["observation.sensor"].shape == ND_FEATURE_SHAPE
    np.testing.assert_array_equal(frame["observation.sensor"], sensor)


def test_build_dataset_frame_scalars_and_images_unchanged():
    features = hw_to_dataset_features(_hw_observation_features(), OBS_STR, use_video=False)
    image = np.zeros(CAMERA_SHAPE, dtype=np.uint8)
    values = {"joint.pos": 1.0, "cam": image, "sensor": np.zeros(ND_FEATURE_SHAPE, dtype=np.float32)}

    frame = build_dataset_frame(features, values, OBS_STR)

    assert set(frame) == {"observation.state", "observation.images.cam", "observation.sensor"}
    np.testing.assert_array_equal(frame["observation.state"], np.array([1.0], dtype=np.float32))
    assert frame["observation.images.cam"] is image


# ── Integration with LeRobotDataset ───────────────────────────────────


@pytest.mark.dataset
def test_validate_frame_accepts_nd_numeric_frame():
    """A frame built from the generated features passes ``validate_frame``."""
    pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
    from lerobot.datasets.feature_utils import validate_frame

    features = hw_to_dataset_features(_hw_observation_features(), OBS_STR, use_video=True)
    values = {
        "joint.pos": 1.0,
        "cam": np.zeros(CAMERA_SHAPE, dtype=np.uint8),
        "sensor": np.zeros(ND_FEATURE_SHAPE, dtype=np.float32),
    }
    frame = build_dataset_frame(features, values, OBS_STR)
    frame["task"] = "task"

    validate_frame(frame, features)


@pytest.mark.dataset
def test_dataset_round_trip_with_nd_numeric_feature(tmp_path):
    """Create, record, save, finalize and reload a dataset holding an (N, C) feature."""
    pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = "dummy/nd_numeric_feature"
    root = tmp_path / "dataset"
    n_frames = 3

    features = hw_to_dataset_features(_hw_observation_features(), OBS_STR, use_video=False)
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=10, features=features, root=root, use_videos=False)

    rng = np.random.default_rng(0)
    sensor_values = [rng.standard_normal(ND_FEATURE_SHAPE).astype(np.float32) for _ in range(n_frames)]
    for sensor in sensor_values:
        values = {"joint.pos": 1.0, "cam": np.zeros(CAMERA_SHAPE, dtype=np.uint8), "sensor": sensor}
        frame = build_dataset_frame(features, values, OBS_STR)
        dataset.add_frame({**frame, "task": "task"})
    dataset.save_episode()
    dataset.finalize()

    reloaded = LeRobotDataset(repo_id, root=root)

    assert reloaded.num_frames == n_frames
    assert reloaded.meta.features["observation.sensor"]["dtype"] == "float32"
    assert tuple(reloaded.meta.features["observation.sensor"]["shape"]) == ND_FEATURE_SHAPE
    for idx, sensor in enumerate(sensor_values):
        item = reloaded[idx]["observation.sensor"]
        assert item.dtype == torch.float32
        assert tuple(item.shape) == ND_FEATURE_SHAPE
        np.testing.assert_array_equal(item.numpy(), sensor)
    assert "observation.sensor" in reloaded.meta.stats


def test_nd_state_does_not_silently_replace_joints():
    with pytest.raises(ValueError, match="collision"):
        hw_to_dataset_features(
            {"joint.pos": float, "state": PolicyFeature(type=FeatureType.STATE, shape=(2, 3))}, "observation"
        )


def test_nd_camera_name_does_not_silently_replace_sensor():
    with pytest.raises(ValueError, match="collision"):
        hw_to_dataset_features(
            {"cam": (8, 8, 3), "images.cam": PolicyFeature(type=FeatureType.STATE, shape=(2, 3))},
            "observation",
        )


def test_standalone_nd_state_remains_supported():
    features = hw_to_dataset_features(
        {"state": PolicyFeature(type=FeatureType.STATE, shape=(2, 3))}, "observation"
    )
    frame = build_dataset_frame(features, {"state": np.ones((2, 3))}, "observation")
    assert frame["observation.state"].shape == (2, 3)
