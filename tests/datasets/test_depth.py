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
"""Tests for depth image functionality."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from lerobot.configs.types import FeatureType
from lerobot.datasets.compute_stats import (
    compute_episode_stats,
    get_feature_stats,
    sample_depth_images,
)
from lerobot.datasets.image_writer import (
    image_array_to_pil_image,
    write_image,
)
from lerobot.datasets.utils import (
    combine_feature_dicts,
    dataset_to_policy_features,
    get_hf_features_from_features,
    hf_transform_to_torch,
    load_depth_as_numpy,
    validate_feature_image_or_video,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def depth_array_hwc():
    """Depth array in (H, W, 1) format, float32 meters."""
    return np.random.uniform(0.5, 5.0, (480, 640, 1)).astype(np.float32)


@pytest.fixture
def depth_array_chw():
    """Depth array in (1, H, W) format, float32 meters."""
    return np.random.uniform(0.5, 5.0, (1, 480, 640)).astype(np.float32)


@pytest.fixture
def depth_array_uint16():
    """Depth array in uint16 mm format."""
    return np.random.randint(500, 5000, (1, 480, 640), dtype=np.uint16)


@pytest.fixture
def depth_feature():
    """Standard depth feature definition."""
    return {
        "observation.images.depth": {
            "dtype": "depth",
            "shape": (1, 480, 640),
            "names": ["channel", "height", "width"],
        }
    }


@pytest.fixture
def depth_feature_hwc():
    """Depth feature with HWC shape convention."""
    return {
        "observation.images.depth": {
            "dtype": "depth",
            "shape": (480, 640, 1),
            "names": ["height", "width", "channel"],
        }
    }


# ============================================================================
# Tests for image_writer.py - Depth image encoding
# ============================================================================


class TestDepthImageEncoding:
    """Tests for encoding depth images to PNG."""

    def test_depth_array_to_pil_chw_format(self, depth_array_chw):
        """Test converting CHW depth array to PIL image."""
        result = image_array_to_pil_image(depth_array_chw, is_depth=True)
        assert isinstance(result, Image.Image)
        assert result.size == (640, 480)  # PIL size is (width, height)
        assert result.mode == "I;16"

    def test_depth_array_to_pil_hwc_format(self, depth_array_hwc):
        """Test converting HWC depth array to PIL image."""
        result = image_array_to_pil_image(depth_array_hwc, is_depth=True)
        assert isinstance(result, Image.Image)
        assert result.size == (640, 480)
        assert result.mode == "I;16"

    def test_depth_array_to_pil_uint16(self, depth_array_uint16):
        """Test that uint16 arrays are preserved without conversion."""
        result = image_array_to_pil_image(depth_array_uint16, is_depth=True)
        assert isinstance(result, Image.Image)
        assert result.mode == "I;16"

    def test_depth_array_to_pil_meters_to_mm_conversion(self):
        """Test that float meters are converted to uint16 mm."""
        # Create depth at exactly 1 meter
        depth_meters = np.ones((1, 100, 100), dtype=np.float32)
        result = image_array_to_pil_image(depth_meters, is_depth=True)

        # Convert back and check value
        result_array = np.array(result)
        assert result_array.dtype == np.uint16
        np.testing.assert_array_equal(result_array, 1000)  # 1m = 1000mm

    def test_depth_array_invalid_channels(self):
        """Test that 3-channel array raises error when is_depth=True."""
        rgb_array = np.random.rand(3, 100, 100).astype(np.float32)
        with pytest.raises(ValueError, match="Depth image must have 1 channel"):
            image_array_to_pil_image(rgb_array, is_depth=True)

    def test_depth_array_negative_values_raises(self):
        """Test that negative depth values raise error."""
        negative_depth = np.array([[[-0.5]]], dtype=np.float32)
        negative_depth = np.broadcast_to(negative_depth, (1, 100, 100)).copy()
        with pytest.raises(ValueError, match="Depth values must be non-negative"):
            image_array_to_pil_image(negative_depth, is_depth=True)

    def test_write_depth_image_roundtrip(self, tmp_path, depth_array_chw):
        """Test full roundtrip: write depth image and read it back."""
        fpath = tmp_path / "test_depth.png"
        write_image(depth_array_chw, fpath, is_depth=True)

        assert fpath.exists()

        # Load back and verify
        loaded = load_depth_as_numpy(fpath, dtype=np.float32, channel_first=True)
        assert loaded.shape == depth_array_chw.shape

        # Check accuracy (should be ~1mm precision due to uint16)
        diff = np.abs(depth_array_chw - loaded)
        assert diff.max() < 0.001  # Less than 1mm error


# ============================================================================
# Tests for utils.py - Depth loading and feature handling
# ============================================================================


class TestDepthLoading:
    """Tests for loading depth images."""

    def test_load_depth_as_numpy_channel_first(self, tmp_path):
        """Test loading depth image in channel-first format."""
        # Create and save a test depth image
        depth_mm = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
        img = Image.fromarray(depth_mm)
        fpath = tmp_path / "depth.png"
        img.save(fpath)

        # Load and verify
        loaded = load_depth_as_numpy(fpath, dtype=np.float32, channel_first=True)
        assert loaded.shape == (1, 2, 2)
        np.testing.assert_allclose(loaded, [[[1.0, 2.0], [3.0, 4.0]]], rtol=1e-3)

    def test_load_depth_as_numpy_channel_last(self, tmp_path):
        """Test loading depth image in channel-last format."""
        depth_mm = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
        img = Image.fromarray(depth_mm)
        fpath = tmp_path / "depth.png"
        img.save(fpath)

        loaded = load_depth_as_numpy(fpath, dtype=np.float32, channel_first=False)
        assert loaded.shape == (2, 2, 1)


class TestDepthFeatureHandling:
    """Tests for depth feature handling in utils.py."""

    def test_get_hf_features_handles_depth(self, depth_feature):
        """Test that depth dtype maps to HuggingFace Image type."""
        from datasets import Image as HFImage

        hf_features = get_hf_features_from_features(depth_feature)
        assert "observation.images.depth" in hf_features
        assert isinstance(hf_features["observation.images.depth"], HFImage)

    def test_dataset_to_policy_features_depth_is_visual(self, depth_feature):
        """Test that depth features are classified as VISUAL type."""
        policy_features = dataset_to_policy_features(depth_feature)
        assert "observation.images.depth" in policy_features
        assert policy_features["observation.images.depth"].type == FeatureType.VISUAL
        assert policy_features["observation.images.depth"].shape == (1, 480, 640)

    def test_validate_feature_depth_valid_chw(self):
        """Test validation of CHW depth array."""
        depth = np.random.rand(1, 480, 640).astype(np.float32)
        error = validate_feature_image_or_video("test", (1, 480, 640), depth)
        assert error == ""

    def test_validate_feature_depth_valid_hwc(self):
        """Test validation of HWC depth array."""
        depth = np.random.rand(480, 640, 1).astype(np.float32)
        error = validate_feature_image_or_video("test", (1, 480, 640), depth)
        assert error == ""

    def test_validate_feature_depth_wrong_channels(self):
        """Test validation fails for wrong number of channels."""
        depth = np.random.rand(3, 480, 640).astype(np.float32)
        error = validate_feature_image_or_video("test", (1, 480, 640), depth)
        assert "does not have the expected shape" in error

    def test_combine_feature_dicts_depth_not_merged(self):
        """Test that depth features are not merged like vectors."""
        g1 = {
            "observation.images.depth": {
                "dtype": "depth",
                "shape": (1, 480, 640),
                "names": ["channel", "height", "width"],
            }
        }
        g2 = {
            "observation.images.depth": {
                "dtype": "depth",
                "shape": (1, 720, 1280),
                "names": ["channel", "height", "width"],
            }
        }
        out = combine_feature_dicts(g1, g2)
        # Last one wins for non-vector types
        assert out["observation.images.depth"]["shape"] == (1, 720, 1280)


class TestHfTransformToTorchDepth:
    """Tests for hf_transform_to_torch with depth images."""

    def test_depth_pil_to_tensor_conversion(self, depth_feature):
        """Test that depth PIL images are converted correctly to tensors."""
        # Create a uint16 depth image
        depth_mm = np.random.randint(500, 5000, (480, 640), dtype=np.uint16)
        depth_pil = Image.fromarray(depth_mm)

        items = {"observation.images.depth": [depth_pil]}
        result = hf_transform_to_torch(items, features=depth_feature)

        tensor = result["observation.images.depth"][0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 480, 640)
        assert tensor.dtype == torch.float32

        # Check conversion to meters
        expected_meters = depth_mm.astype(np.float32) / 1000.0
        np.testing.assert_allclose(tensor.numpy().squeeze(), expected_meters, rtol=1e-5)

    def test_depth_not_normalized_to_0_1(self, depth_feature):
        """Test that depth values are NOT normalized to [0,1] like RGB."""
        # 5 meters = 5000mm
        depth_mm = np.full((480, 640), 5000, dtype=np.uint16)
        depth_pil = Image.fromarray(depth_mm)

        items = {"observation.images.depth": [depth_pil]}
        result = hf_transform_to_torch(items, features=depth_feature)

        tensor = result["observation.images.depth"][0]
        # Should be 5.0 meters, not normalized to [0,1]
        assert tensor.max().item() == pytest.approx(5.0, rel=1e-3)


# ============================================================================
# Tests for compute_stats.py - Depth statistics
# ============================================================================


class TestDepthStatistics:
    """Tests for depth image statistics computation."""

    def test_get_feature_stats_depth_images(self):
        """Test computing statistics for depth images."""
        # Depth data in meters: (batch, channels, height, width)
        data = np.random.uniform(0.5, 5.0, (100, 1, 32, 32)).astype(np.float32)
        stats = get_feature_stats(data, axis=(0, 2, 3), keepdims=True)

        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "count" in stats
        np.testing.assert_equal(stats["count"], np.array([100]))

        # Stats should have shape (1, 1, 1, 1) with keepdims=True
        assert stats["min"].shape == (1, 1, 1, 1)

    def test_sample_depth_images(self, tmp_path):
        """Test sampling depth images from paths."""
        # Create test depth images
        paths = []
        for i in range(10):
            depth_mm = np.random.randint(500, 5000, (100, 100), dtype=np.uint16)
            img = Image.fromarray(depth_mm)
            fpath = tmp_path / f"depth_{i}.png"
            img.save(fpath)
            paths.append(str(fpath))

        images = sample_depth_images(paths)
        assert isinstance(images, np.ndarray)
        assert images.dtype == np.float32
        assert images.shape[1] == 1  # Single channel

    def test_compute_episode_stats_with_depth(self, tmp_path):
        """Test computing episode stats that include depth features."""

        def mock_load_depth(path, dtype, channel_first):
            return np.ones((1, 32, 32), dtype=dtype) * 2.5  # 2.5 meters

        # Create mock depth image paths
        depth_paths = [str(tmp_path / f"depth_{i}.png") for i in range(10)]

        episode_data = {
            "observation.images.depth": depth_paths,
            "action": np.random.rand(10, 6).astype(np.float32),
        }
        features = {
            "observation.images.depth": {"dtype": "depth", "shape": (1, 32, 32)},
            "action": {"dtype": "float32", "shape": (6,)},
        }

        with patch("lerobot.datasets.compute_stats.load_depth_as_numpy", side_effect=mock_load_depth):
            stats = compute_episode_stats(episode_data, features)

        # Depth stats should NOT be divided by 255
        assert "observation.images.depth" in stats
        depth_stats = stats["observation.images.depth"]
        # Mean should be around 2.5 meters (not normalized)
        np.testing.assert_allclose(depth_stats["mean"], 2.5, rtol=0.1)


# ============================================================================
# Tests for mixed RGB and depth features
# ============================================================================


class TestMixedRgbDepthFeatures:
    """Tests for datasets with both RGB and depth features."""

    def test_policy_features_mixed(self):
        """Test converting mixed RGB and depth features to policy features."""
        features = {
            "observation.images.rgb": {
                "dtype": "image",
                "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            },
            "observation.images.depth": {
                "dtype": "depth",
                "shape": (1, 480, 640),
                "names": ["channel", "height", "width"],
            },
        }
        policy_features = dataset_to_policy_features(features)

        assert policy_features["observation.images.rgb"].type == FeatureType.VISUAL
        assert policy_features["observation.images.rgb"].shape == (3, 480, 640)

        assert policy_features["observation.images.depth"].type == FeatureType.VISUAL
        assert policy_features["observation.images.depth"].shape == (1, 480, 640)

    def test_hf_transform_mixed_rgb_depth(self):
        """Test HF transform with both RGB and depth images."""
        features = {
            "observation.images.rgb": {"dtype": "image", "shape": (3, 480, 640)},
            "observation.images.depth": {"dtype": "depth", "shape": (1, 480, 640)},
        }

        # Create test images
        rgb_pil = Image.new("RGB", (640, 480), color=(128, 128, 128))
        depth_mm = np.full((480, 640), 2000, dtype=np.uint16)
        depth_pil = Image.fromarray(depth_mm)

        items = {
            "observation.images.rgb": [rgb_pil],
            "observation.images.depth": [depth_pil],
        }
        result = hf_transform_to_torch(items, features=features)

        # RGB should be normalized to [0, 1]
        rgb_tensor = result["observation.images.rgb"][0]
        assert rgb_tensor.max() <= 1.0
        assert rgb_tensor.shape == (3, 480, 640)

        # Depth should be in meters
        depth_tensor = result["observation.images.depth"][0]
        assert depth_tensor.max().item() == pytest.approx(2.0, rel=1e-3)  # 2000mm = 2m
        assert depth_tensor.shape == (1, 480, 640)
