# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import time

import numpy as np
import pytest

pytest.importorskip("grpc")

from lerobot.async_inference.compression import (  # noqa: E402
    CompressedImage,
    compress_image,
    compress_raw_observation,
    compress_timed_observation,
    compression_stats,
    decompress_image,
    decompress_timed_observation,
    image_keys_from_lerobot_features,
)
from lerobot.async_inference.configs import RobotClientConfig  # noqa: E402
from lerobot.async_inference.helpers import TimedObservation  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402
from tests.mocks.mock_robot import MockRobotConfig  # noqa: E402


def _structured_image(height: int = 96, width: int = 128) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    average = ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2).astype(np.uint8)
    return np.stack([xx, yy, average], axis=-1)


def test_image_keys_from_lerobot_features_extracts_raw_camera_keys():
    lerobot_features = {
        OBS_STATE: {"dtype": "float32", "shape": [2], "names": ["joint_1", "joint_2"]},
        f"{OBS_IMAGES}.front": {"dtype": "image", "shape": [96, 128, 3]},
        f"{OBS_IMAGES}.wrist": {"dtype": "image", "shape": [64, 64, 3]},
    }

    assert image_keys_from_lerobot_features(lerobot_features) == {"front", "wrist"}


def test_png_round_trip_is_lossless_for_uint8_images():
    image = _structured_image()

    compressed = compress_image(image, "png", quality=90)
    decompressed = decompress_image(compressed)

    assert decompressed.dtype == image.dtype
    assert decompressed.shape == image.shape
    np.testing.assert_array_equal(decompressed, image)


def test_jpeg_round_trip_preserves_shape_dtype_and_reduces_structured_payload():
    image = _structured_image()

    compressed = compress_image(image, "jpeg", quality=95)
    decompressed = decompress_image(compressed)

    assert decompressed.dtype == image.dtype
    assert decompressed.shape == image.shape
    assert len(compressed.data) < image.nbytes
    mean_abs_error = np.abs(decompressed.astype(np.int16) - image.astype(np.int16)).mean()
    assert mean_abs_error < 5


def test_compress_raw_observation_preserves_non_image_fields_and_does_not_mutate_input():
    image = _structured_image()
    raw_observation = {
        "joint_1": 1.0,
        "joint_2": 2.0,
        "front": image,
        "task": "pick up the cube",
    }

    compressed = compress_raw_observation(raw_observation, {"front"}, "jpeg", quality=90)

    assert compressed is not raw_observation
    assert raw_observation["front"] is image
    assert compressed["joint_1"] == raw_observation["joint_1"]
    assert compressed["joint_2"] == raw_observation["joint_2"]
    assert compressed["task"] == raw_observation["task"]
    assert isinstance(compressed["front"], CompressedImage)


def test_compress_timed_observation_preserves_timing_metadata():
    timestamp = time.time()
    observation = TimedObservation(
        timestamp=timestamp,
        timestep=12,
        observation={"front": _structured_image(), "joint_1": 1.0},
        must_go=True,
    )

    compressed = compress_timed_observation(observation, {"front"}, "png", quality=90)
    stats = compression_stats(compressed.get_observation())
    decompressed = decompress_timed_observation(compressed)

    assert compressed is not observation
    assert stats.image_count == 1
    assert decompressed.get_timestamp() == timestamp
    assert decompressed.get_timestep() == 12
    assert decompressed.must_go is True
    np.testing.assert_array_equal(decompressed.get_observation()["front"], observation.get_observation()["front"])
    assert decompressed.get_observation()["joint_1"] == 1.0


@pytest.mark.parametrize(
    ("image", "match"),
    [
        (np.zeros((4, 4), dtype=np.float32), "uint8"),
        (np.zeros((4,), dtype=np.uint8), "2D or 3D"),
        (np.zeros((4, 4, 2), dtype=np.uint8), "1-, 3-, or 4-channel"),
    ],
)
def test_compress_image_rejects_unsupported_image_payloads(image, match):
    with pytest.raises(ValueError, match=match):
        compress_image(image, "jpeg", quality=90)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observation_image_compression": "webp"},
        {"observation_image_compression_quality": 0},
        {"observation_image_compression_quality": 101},
    ],
)
def test_robot_client_config_validates_observation_compression(kwargs):
    config_kwargs = {
        "policy_type": "act",
        "pretrained_name_or_path": "test/model",
        "robot": MockRobotConfig(),
        "actions_per_chunk": 1,
    }
    config_kwargs.update(kwargs)

    with pytest.raises(ValueError):
        RobotClientConfig(**config_kwargs)
