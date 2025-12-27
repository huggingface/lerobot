#!/usr/bin/env python

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

import pytest
import torch

from lerobot.configs.types import (
    FeatureType,
    PipelineFeatureType,
    PolicyFeature,
)
from lerobot.processor.env_processor import IsaaclabArenaProcessorStep
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, OBS_STR

# Test constants
BATCH_SIZE = 2
STATE_DIM = 16
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Generic test keys (not real robot keys)
TEST_STATE_KEY = "test_state_obs"
TEST_CAMERA_KEY = "test_rgb_cam"


@pytest.fixture
def processor():
    """Default processor with test keys."""
    return IsaaclabArenaProcessorStep(
        state_keys=(TEST_STATE_KEY,),
        camera_keys=(TEST_CAMERA_KEY,),
    )


@pytest.fixture
def sample_observation():
    """Sample IsaacLab Arena observation with state and camera data."""
    return {
        f"{OBS_STR}.policy": {
            TEST_STATE_KEY: torch.randn(BATCH_SIZE, STATE_DIM),
        },
        f"{OBS_STR}.camera_obs": {
            TEST_CAMERA_KEY: torch.randint(0, 255, (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), dtype=torch.uint8),
        },
    }


# =============================================================================
# State Processing Tests
# =============================================================================


def test_state_extraction(processor, sample_observation):
    """Test that state is extracted and converted to float32."""
    processed = processor.observation(sample_observation)

    assert OBS_STATE in processed
    assert processed[OBS_STATE].shape == (BATCH_SIZE, STATE_DIM)
    assert processed[OBS_STATE].dtype == torch.float32


def test_state_concatenation_multiple_keys():
    """Test that multiple state keys are concatenated in order."""
    dim1, dim2 = 10, 6
    processor = IsaaclabArenaProcessorStep(
        state_keys=("state_alpha", "state_beta"),
        camera_keys=(),
    )

    obs = {
        f"{OBS_STR}.policy": {
            "state_alpha": torch.ones(BATCH_SIZE, dim1),
            "state_beta": torch.ones(BATCH_SIZE, dim2) * 2,
        },
    }

    processed = processor.observation(obs)

    state = processed[OBS_STATE]
    assert state.shape == (BATCH_SIZE, dim1 + dim2)
    # Verify ordering: first dim1 elements are 1s, last dim2 are 2s
    assert torch.all(state[:, :dim1] == 1.0)
    assert torch.all(state[:, dim1:] == 2.0)


def test_state_flattening_higher_dims():
    """Test that state with dim > 2 is flattened to (B, -1)."""
    processor = IsaaclabArenaProcessorStep(
        state_keys=("multidim_state",),
        camera_keys=(),
    )

    # Shape (B, 4, 4) -> should flatten to (B, 16)
    obs = {
        f"{OBS_STR}.policy": {
            "multidim_state": torch.randn(BATCH_SIZE, 4, 4),
        },
    }

    processed = processor.observation(obs)

    assert processed[OBS_STATE].shape == (BATCH_SIZE, 16)


def test_state_filters_to_configured_keys():
    """Test that only configured state_keys are extracted."""
    processor = IsaaclabArenaProcessorStep(
        state_keys=("included_key",),
        camera_keys=(),
    )

    obs = {
        f"{OBS_STR}.policy": {
            "included_key": torch.randn(BATCH_SIZE, 10),
            "excluded_key": torch.randn(BATCH_SIZE, 6),  # Should be ignored
        },
    }

    processed = processor.observation(obs)

    # Only included_key (dim 10) should be included
    assert processed[OBS_STATE].shape == (BATCH_SIZE, 10)


def test_missing_state_key_skipped():
    """Test that missing state keys in observation are skipped."""
    processor = IsaaclabArenaProcessorStep(
        state_keys=("present_key", "missing_key"),
        camera_keys=(),
    )

    obs = {
        f"{OBS_STR}.policy": {
            "present_key": torch.randn(BATCH_SIZE, 10),
            # missing_key not present
        },
    }

    processed = processor.observation(obs)

    # Should only have present_key
    assert processed[OBS_STATE].shape == (BATCH_SIZE, 10)


# =============================================================================
# Camera/Image Processing Tests
# =============================================================================


def test_camera_permutation_bhwc_to_bchw(processor, sample_observation):
    """Test images are permuted from (B, H, W, C) to (B, C, H, W)."""
    processed = processor.observation(sample_observation)

    img_key = f"{OBS_IMAGES}.{TEST_CAMERA_KEY}"
    assert img_key in processed
    img = processed[img_key]
    assert img.shape == (BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH)


def test_camera_uint8_to_normalized_float32(processor):
    """Test that uint8 images are normalized to float32 [0, 1]."""
    obs = {
        f"{OBS_STR}.camera_obs": {
            TEST_CAMERA_KEY: torch.full((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), 255, dtype=torch.uint8),
        },
    }

    processed = processor.observation(obs)

    img = processed[f"{OBS_IMAGES}.{TEST_CAMERA_KEY}"]
    assert img.dtype == torch.float32
    assert torch.allclose(img, torch.ones_like(img))


def test_camera_float32_passthrough(processor):
    """Test that float32 images are kept as float32."""
    original_img = torch.rand(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3, dtype=torch.float32)
    obs = {
        f"{OBS_STR}.camera_obs": {
            TEST_CAMERA_KEY: original_img.clone(),
        },
    }

    processed = processor.observation(obs)

    img = processed[f"{OBS_IMAGES}.{TEST_CAMERA_KEY}"]
    assert img.dtype == torch.float32
    # Values should be same (just permuted)
    expected = original_img.permute(0, 3, 1, 2)
    assert torch.allclose(img, expected)


def test_camera_other_dtype_converted_to_float(processor):
    """Test that non-uint8, non-float32 dtypes are converted to float."""
    obs = {
        f"{OBS_STR}.camera_obs": {
            TEST_CAMERA_KEY: torch.randint(0, 255, (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), dtype=torch.int32),
        },
    }

    processed = processor.observation(obs)

    img = processed[f"{OBS_IMAGES}.{TEST_CAMERA_KEY}"]
    assert img.dtype == torch.float32


def test_camera_filters_to_configured_keys():
    """Test that only configured camera_keys are extracted."""
    processor = IsaaclabArenaProcessorStep(
        state_keys=(),
        camera_keys=("included_cam",),
    )

    obs = {
        f"{OBS_STR}.camera_obs": {
            "included_cam": torch.randint(0, 255, (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), dtype=torch.uint8),
            "excluded_cam": torch.randint(0, 255, (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), dtype=torch.uint8),
        },
    }

    processed = processor.observation(obs)

    assert f"{OBS_IMAGES}.included_cam" in processed
    assert f"{OBS_IMAGES}.excluded_cam" not in processed


def test_camera_key_preserved_exactly():
    """Test that camera key name is used exactly (no suffix stripping)."""
    processor = IsaaclabArenaProcessorStep(
        state_keys=(),
        camera_keys=("my_cam_rgb",),
    )

    obs = {
        f"{OBS_STR}.camera_obs": {
            "my_cam_rgb": torch.randint(0, 255, (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), dtype=torch.uint8),
        },
    }

    processed = processor.observation(obs)

    # Key should be exactly as configured, with _rgb suffix intact
    assert f"{OBS_IMAGES}.my_cam_rgb" in processed
    assert f"{OBS_IMAGES}.my_cam" not in processed


# =============================================================================
# Edge Cases & Missing Data Tests
# =============================================================================


def test_missing_camera_obs_section(processor):
    """Test processor handles observation without camera_obs section."""
    obs = {
        f"{OBS_STR}.policy": {
            TEST_STATE_KEY: torch.randn(BATCH_SIZE, STATE_DIM),
        },
    }

    processed = processor.observation(obs)

    assert OBS_STATE in processed
    assert not any(k.startswith(OBS_IMAGES) for k in processed)


def test_missing_policy_obs_section(processor):
    """Test processor handles observation without policy section."""
    obs = {
        f"{OBS_STR}.camera_obs": {
            TEST_CAMERA_KEY: torch.randint(0, 255, (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), dtype=torch.uint8),
        },
    }

    processed = processor.observation(obs)

    assert f"{OBS_IMAGES}.{TEST_CAMERA_KEY}" in processed
    assert OBS_STATE not in processed


def test_empty_observation(processor):
    """Test processor handles empty observation dict."""
    processed = processor.observation({})

    assert OBS_STATE not in processed
    assert not any(k.startswith(OBS_IMAGES) for k in processed)


def test_no_matching_state_keys():
    """Test processor when no state keys match observation."""
    processor = IsaaclabArenaProcessorStep(
        state_keys=("nonexistent_key",),
        camera_keys=(),
    )

    obs = {
        f"{OBS_STR}.policy": {
            "some_other_key": torch.randn(BATCH_SIZE, STATE_DIM),
        },
    }

    processed = processor.observation(obs)

    # No state because no keys matched
    assert OBS_STATE not in processed


def test_no_matching_camera_keys():
    """Test processor when no camera keys match observation."""
    processor = IsaaclabArenaProcessorStep(
        state_keys=(),
        camera_keys=("nonexistent_cam",),
    )

    obs = {
        f"{OBS_STR}.camera_obs": {
            "some_other_cam": torch.randint(
                0, 255, (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), dtype=torch.uint8
            ),
        },
    }

    processed = processor.observation(obs)

    assert not any(k.startswith(OBS_IMAGES) for k in processed)


# =============================================================================
# Configuration Tests
# =============================================================================


def test_default_keys():
    """Test default state_keys and camera_keys values."""
    processor = IsaaclabArenaProcessorStep()

    assert processor.state_keys == ("robot_joint_pos",)
    assert processor.camera_keys == ("robot_pov_cam_rgb",)


def test_custom_keys_configuration():
    """Test processor with custom state and camera keys."""
    processor = IsaaclabArenaProcessorStep(
        state_keys=("pos_xyz", "quat_wxyz", "grip_val"),
        camera_keys=("front_view", "wrist_view"),
    )

    obs = {
        f"{OBS_STR}.policy": {
            "pos_xyz": torch.randn(BATCH_SIZE, 3),
            "quat_wxyz": torch.randn(BATCH_SIZE, 4),
            "grip_val": torch.randn(BATCH_SIZE, 1),
        },
        f"{OBS_STR}.camera_obs": {
            "front_view": torch.randint(0, 255, (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), dtype=torch.uint8),
            "wrist_view": torch.randint(0, 255, (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), dtype=torch.uint8),
        },
    }

    processed = processor.observation(obs)

    # State should be concatenated: 3 + 4 + 1 = 8
    assert processed[OBS_STATE].shape == (BATCH_SIZE, 8)
    # Both cameras should be present
    assert f"{OBS_IMAGES}.front_view" in processed
    assert f"{OBS_IMAGES}.wrist_view" in processed


# =============================================================================
# transform_features Tests
# =============================================================================


def test_transform_features_passthrough(processor):
    """Test that transform_features returns features unchanged."""
    input_features = {
        PipelineFeatureType.OBSERVATION: {
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(16,),
            ),
            "observation.images.cam": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 64, 64),
            ),
        },
        PipelineFeatureType.ACTION: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
    }

    output_features = processor.transform_features(input_features)

    # Should be unchanged
    assert output_features == input_features
