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

import math
import pickle
import time

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.scripts.server.helpers import (
    FPSTracker,
    TimedAction,
    TimedObservation,
    observations_similar,
    prepare_image,
    prepare_raw_observation,
    raw_observation_to_observation,
    resize_robot_observation_image,
)

# ---------------------------------------------------------------------
# FPSTracker
# ---------------------------------------------------------------------


def test_fps_tracker_first_observation():
    """First observation should initialize timestamp and return 0 FPS."""
    tracker = FPSTracker(target_fps=30.0)
    timestamp = 1000.0

    metrics = tracker.calculate_fps_metrics(timestamp)

    assert tracker.first_timestamp == timestamp
    assert tracker.total_obs_count == 1
    assert metrics["avg_fps"] == 0.0
    assert metrics["target_fps"] == 30.0


def test_fps_tracker_single_interval():
    """Two observations 1 second apart should give 1 FPS."""
    tracker = FPSTracker(target_fps=30.0)

    # First observation at t=0
    metrics1 = tracker.calculate_fps_metrics(0.0)
    assert metrics1["avg_fps"] == 0.0

    # Second observation at t=1 (1 second later)
    metrics2 = tracker.calculate_fps_metrics(1.0)
    expected_fps = 1.0  # (2-1) observations / 1.0 seconds = 1 FPS
    assert math.isclose(metrics2["avg_fps"], expected_fps, rel_tol=1e-6)


def test_fps_tracker_multiple_intervals():
    """Multiple observations should calculate correct average FPS."""
    tracker = FPSTracker(target_fps=30.0)

    # Simulate 5 observations over 2 seconds (should be 2 FPS average)
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0]

    for i, ts in enumerate(timestamps):
        metrics = tracker.calculate_fps_metrics(ts)

        if i == 0:
            assert metrics["avg_fps"] == 0.0
        elif i == len(timestamps) - 1:
            # After 5 observations over 2 seconds: (5-1)/2 = 2 FPS
            expected_fps = 2.0
            assert math.isclose(metrics["avg_fps"], expected_fps, rel_tol=1e-6)


def test_fps_tracker_irregular_intervals():
    """FPS calculation should work with irregular time intervals."""
    tracker = FPSTracker(target_fps=30.0)

    # Irregular timestamps: 0, 0.1, 0.5, 2.0, 3.0 seconds
    timestamps = [0.0, 0.1, 0.5, 2.0, 3.0]

    for ts in timestamps:
        metrics = tracker.calculate_fps_metrics(ts)

    # 5 observations over 3 seconds: (5-1)/3 = 1.333... FPS
    expected_fps = 4.0 / 3.0
    assert math.isclose(metrics["avg_fps"], expected_fps, rel_tol=1e-6)


# ---------------------------------------------------------------------
# TimedData helpers
# ---------------------------------------------------------------------


def test_timed_action_getters():
    """TimedAction stores & returns timestamp, action tensor and timestep."""
    ts = time.time()
    action = torch.arange(10)
    ta = TimedAction(timestamp=ts, action=action, timestep=0)

    assert math.isclose(ta.get_timestamp(), ts, rel_tol=0, abs_tol=1e-6)
    torch.testing.assert_close(ta.get_action(), action)
    assert ta.get_timestep() == 0


def test_timed_observation_getters():
    """TimedObservation stores & returns timestamp, dict and timestep."""
    ts = time.time()
    obs_dict = {"observation.state": torch.ones(6)}
    to = TimedObservation(timestamp=ts, observation=obs_dict, timestep=0)

    assert math.isclose(to.get_timestamp(), ts, rel_tol=0, abs_tol=1e-6)
    assert to.get_observation() is obs_dict
    assert to.get_timestep() == 0


def test_timed_data_deserialization_data_getters():
    """TimedAction / TimedObservation survive a round-trip through ``pickle``.

    The async-inference stack uses ``pickle.dumps`` to move these objects across
    the gRPC boundary (see RobotClient.send_observation and PolicyServer.StreamActions).
    This test ensures that the payload keeps its content intact after
    the (de)serialization round-trip.
    """
    ts = time.time()

    # ------------------------------------------------------------------
    # TimedAction
    # ------------------------------------------------------------------
    original_action = torch.randn(6)
    ta_in = TimedAction(timestamp=ts, action=original_action, timestep=13)

    # Serialize → bytes → deserialize
    ta_bytes = pickle.dumps(ta_in)  # nosec
    ta_out: TimedAction = pickle.loads(ta_bytes)  # nosec B301

    # Identity & content checks
    assert math.isclose(ta_out.get_timestamp(), ts, rel_tol=0, abs_tol=1e-6)
    assert ta_out.get_timestep() == 13
    torch.testing.assert_close(ta_out.get_action(), original_action)

    # ------------------------------------------------------------------
    # TimedObservation
    # ------------------------------------------------------------------
    obs_dict = {"observation.state": torch.arange(4).float()}
    to_in = TimedObservation(timestamp=ts, observation=obs_dict, timestep=7, must_go=True)

    to_bytes = pickle.dumps(to_in)  # nosec
    to_out: TimedObservation = pickle.loads(to_bytes)  # nosec B301

    assert math.isclose(to_out.get_timestamp(), ts, rel_tol=0, abs_tol=1e-6)
    assert to_out.get_timestep() == 7
    assert to_out.must_go is True
    assert to_out.get_observation().keys() == obs_dict.keys()
    torch.testing.assert_close(to_out.get_observation()["observation.state"], obs_dict["observation.state"])


# ---------------------------------------------------------------------
# observations_similar()
# ---------------------------------------------------------------------


def _make_obs(state: torch.Tensor) -> TimedObservation:
    """Create a TimedObservation with raw robot observation format."""
    return TimedObservation(
        timestamp=time.time(),
        observation={
            "shoulder": state[0].item() if len(state) > 0 else 0.0,
            "elbow": state[1].item() if len(state) > 1 else 0.0,
            "wrist": state[2].item() if len(state) > 2 else 0.0,
            "gripper": state[3].item() if len(state) > 3 else 0.0,
        },
        timestep=0,
    )


def test_observations_similar_true():
    """Distance below atol → observations considered similar."""
    # Create mock lerobot features for the similarity check
    lerobot_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [4],
            "names": ["shoulder", "elbow", "wrist", "gripper"],
        }
    }

    obs1 = _make_obs(torch.zeros(4))
    obs2 = _make_obs(0.5 * torch.ones(4))
    assert observations_similar(obs1, obs2, lerobot_features, atol=2.0)

    obs3 = _make_obs(2.0 * torch.ones(4))
    assert not observations_similar(obs1, obs3, lerobot_features, atol=2.0)


# ---------------------------------------------------------------------
# raw_observation_to_observation and helpers
# ---------------------------------------------------------------------


def _create_mock_robot_observation():
    """Create a mock robot observation with motor positions and camera images."""
    return {
        "shoulder": 1.0,
        "elbow": 2.0,
        "wrist": 3.0,
        "gripper": 0.5,
        "laptop": np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8),
        "phone": np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8),
    }


def _create_mock_lerobot_features():
    """Create mock lerobot features mapping similar to what hw_to_dataset_features returns."""
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": [4],
            "names": ["shoulder", "elbow", "wrist", "gripper"],
        },
        "observation.images.laptop": {
            "dtype": "image",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.images.phone": {
            "dtype": "image",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        },
    }


def _create_mock_policy_image_features():
    """Create mock policy image features with different resolutions."""
    return {
        "observation.images.laptop": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),  # Policy expects smaller resolution
        ),
        "observation.images.phone": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 160, 160),  # Different resolution for second camera
        ),
    }


def test_prepare_image():
    """Test image preprocessing: int8 → float32, normalization to [0,1]."""
    # Create mock int8 image data
    image_int8 = torch.randint(0, 256, size=(3, 224, 224), dtype=torch.uint8)

    processed = prepare_image(image_int8)

    # Check dtype conversion
    assert processed.dtype == torch.float32

    # Check normalization range
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0

    # Check that values are scaled correctly (255 → 1.0, 0 → 0.0)
    if image_int8.max() == 255:
        assert torch.isclose(processed.max(), torch.tensor(1.0), atol=1e-6)
    if image_int8.min() == 0:
        assert torch.isclose(processed.min(), torch.tensor(0.0), atol=1e-6)

    # Check memory contiguity
    assert processed.is_contiguous()


def test_resize_robot_observation_image():
    """Test image resizing from robot resolution to policy resolution."""
    # Create mock image: (H=480, W=640, C=3)
    original_image = torch.randint(0, 256, size=(480, 640, 3), dtype=torch.uint8)
    target_shape = (3, 224, 224)  # (C, H, W)

    resized = resize_robot_observation_image(original_image, target_shape)

    # Check output shape matches target
    assert resized.shape == target_shape

    # Check that original image had different dimensions
    assert original_image.shape != resized.shape

    # Check that resizing preserves value range
    assert resized.min() >= 0
    assert resized.max() <= 255


def test_prepare_raw_observation():
    """Test the preparation of raw robot observation to lerobot format."""
    robot_obs = _create_mock_robot_observation()
    lerobot_features = _create_mock_lerobot_features()
    policy_image_features = _create_mock_policy_image_features()

    prepared = prepare_raw_observation(robot_obs, lerobot_features, policy_image_features)

    # Check that state is properly extracted and batched
    assert "observation.state" in prepared
    state = prepared["observation.state"]
    assert isinstance(state, torch.Tensor)
    assert state.shape == (1, 4)  # Batched state

    # Check that images are processed and resized
    assert "observation.images.laptop" in prepared
    assert "observation.images.phone" in prepared

    laptop_img = prepared["observation.images.laptop"]
    phone_img = prepared["observation.images.phone"]

    # Check image shapes match policy requirements
    assert laptop_img.shape == policy_image_features["observation.images.laptop"].shape
    assert phone_img.shape == policy_image_features["observation.images.phone"].shape

    # Check that images are tensors
    assert isinstance(laptop_img, torch.Tensor)
    assert isinstance(phone_img, torch.Tensor)


def test_raw_observation_to_observation_basic():
    """Test the main raw_observation_to_observation function."""
    robot_obs = _create_mock_robot_observation()
    lerobot_features = _create_mock_lerobot_features()
    policy_image_features = _create_mock_policy_image_features()
    device = "cpu"

    observation = raw_observation_to_observation(robot_obs, lerobot_features, policy_image_features, device)

    # Check that all expected keys are present
    assert "observation.state" in observation
    assert "observation.images.laptop" in observation
    assert "observation.images.phone" in observation

    # Check state processing
    state = observation["observation.state"]
    assert isinstance(state, torch.Tensor)
    assert state.device.type == device
    assert state.shape == (1, 4)  # Batched

    # Check image processing
    laptop_img = observation["observation.images.laptop"]
    phone_img = observation["observation.images.phone"]

    # Images should have batch dimension: (B, C, H, W)
    assert laptop_img.shape == (1, 3, 224, 224)
    assert phone_img.shape == (1, 3, 160, 160)

    # Check device placement
    assert laptop_img.device.type == device
    assert phone_img.device.type == device

    # Check image dtype and range (should be float32 in [0, 1])
    assert laptop_img.dtype == torch.float32
    assert phone_img.dtype == torch.float32
    assert laptop_img.min() >= 0.0 and laptop_img.max() <= 1.0
    assert phone_img.min() >= 0.0 and phone_img.max() <= 1.0


def test_raw_observation_to_observation_with_non_tensor_data():
    """Test that non-tensor data (like task strings) is preserved."""
    robot_obs = _create_mock_robot_observation()
    robot_obs["task"] = "pick up the red cube"  # Add string instruction

    lerobot_features = _create_mock_lerobot_features()
    policy_image_features = _create_mock_policy_image_features()
    device = "cpu"

    observation = raw_observation_to_observation(robot_obs, lerobot_features, policy_image_features, device)

    # Check that task string is preserved
    assert "task" in observation
    assert observation["task"] == "pick up the red cube"
    assert isinstance(observation["task"], str)


@torch.no_grad()
def test_raw_observation_to_observation_device_handling():
    """Test that tensors are properly moved to the specified device."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    robot_obs = _create_mock_robot_observation()
    lerobot_features = _create_mock_lerobot_features()
    policy_image_features = _create_mock_policy_image_features()

    observation = raw_observation_to_observation(robot_obs, lerobot_features, policy_image_features, device)

    # Check that all tensors are on the correct device
    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            assert value.device.type == device, f"Tensor {key} not on {device}"


def test_raw_observation_to_observation_deterministic():
    """Test that the function produces consistent results for the same input."""
    robot_obs = _create_mock_robot_observation()
    lerobot_features = _create_mock_lerobot_features()
    policy_image_features = _create_mock_policy_image_features()
    device = "cpu"

    # Run twice with same input
    obs1 = raw_observation_to_observation(robot_obs, lerobot_features, policy_image_features, device)
    obs2 = raw_observation_to_observation(robot_obs, lerobot_features, policy_image_features, device)

    # Results should be identical
    assert set(obs1.keys()) == set(obs2.keys())

    for key in obs1:
        if isinstance(obs1[key], torch.Tensor):
            torch.testing.assert_close(obs1[key], obs2[key])
        else:
            assert obs1[key] == obs2[key]


def test_image_processing_pipeline_preserves_content():
    """Test that the image processing pipeline preserves recognizable patterns."""
    # Create an image with a specific pattern
    original_img = np.zeros((100, 100, 3), dtype=np.uint8)
    original_img[25:75, 25:75, :] = 255  # White square in center

    robot_obs = {"shoulder": 1.0, "elbow": 1.0, "wrist": 1.0, "gripper": 1.0, "laptop": original_img}
    lerobot_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [4],
            "names": ["shoulder", "elbow", "wrist", "gripper"],
        },
        "observation.images.laptop": {
            "dtype": "image",
            "shape": [100, 100, 3],
            "names": ["height", "width", "channels"],
        },
    }
    policy_image_features = {
        "observation.images.laptop": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 50, 50),  # Downsamples from 100x100
        )
    }

    observation = raw_observation_to_observation(robot_obs, lerobot_features, policy_image_features, "cpu")

    processed_img = observation["observation.images.laptop"].squeeze(0)  # Remove batch dim

    # Check that the center region has higher values than corners
    # Due to bilinear interpolation, exact values will change but pattern should remain
    center_val = processed_img[:, 25, 25].mean()  # Center of 50x50 image
    corner_val = processed_img[:, 5, 5].mean()  # Corner

    assert center_val > corner_val, "Image processing should preserve recognizable patterns"
