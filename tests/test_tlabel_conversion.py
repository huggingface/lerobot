"""Tests for TLabel to LeRobot conversion."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "tlabel"))

from convert_tlabel_to_lerobot import (
    DEFAULT_TACTILE_FEATURES,
    SENSOR_CONFIGS,
    build_features,
    extract_tactile_features,
    detect_image_dimensions,
    _load_manual,
    _load_csv_episodes,
)


# ──────────────────────────────────────────────────────────────────────
# Feature building tests (no tlabel/lerobot dependency)
# ──────────────────────────────────────────────────────────────────────
class TestFeatureBuilding:
    """Test feature dict construction."""

    def test_default_features(self):
        """Default features include all 8 tactile groups + base features."""
        features = build_features("gelsight")
        assert "timestamp" in features
        assert "episode_index" in features
        assert "frame_index" in features
        assert "index" in features
        assert "task_index" in features
        assert "observation.tactile.contact" in features
        assert "observation.tactile.force" in features
        assert "observation.tactile.deformation" in features
        assert "observation.tactile.slip" in features
        assert "observation.tactile.texture" in features
        assert "observation.tactile.contact_geometry" in features
        assert "observation.tactile.field" in features
        assert "observation.tactile.dynamics" in features

    def test_gelsight_has_image(self):
        """GelSight sensor includes tactile image feature."""
        features = build_features("gelsight", has_image=True)
        assert "observation.images.tactile" in features
        assert features["observation.images.tactile"]["dtype"] == "video"

    def test_gelsight_default_image_shape(self):
        """Default image shape is 480x640 when no image_shape provided."""
        features = build_features("gelsight", has_image=True)
        assert features["observation.images.tactile"]["shape"] == [480, 640, 3]

    def test_gelsight_custom_image_shape(self):
        """Image shape is correctly set when image_shape is provided."""
        features = build_features("gelsight", has_image=True, image_shape=(240, 320))
        assert features["observation.images.tactile"]["shape"] == [240, 320, 3]

    def test_paxini_no_image(self):
        """PaXini sensor does not include image feature."""
        features = build_features("paxini", has_image=False)
        assert "observation.images.tactile" not in features

    def test_custom_config(self, tmp_path):
        """Custom YAML config overrides defaults."""
        config = tmp_path / "custom.yaml"
        config.write_text(
            "observation.tactile.contact:\n"
            "  dtype: float32\n"
            "  shape: [1]\n"
            "  names: [contact]\n"
        )
        features = build_features("gelsight", config_path=str(config))
        assert "observation.tactile.contact" in features


# ──────────────────────────────────────────────────────────────────────
# Feature extraction tests (no tlabel/lerobot dependency)
# ──────────────────────────────────────────────────────────────────────
class TestFeatureExtraction:
    """Test TLabel frame to LeRobot feature extraction."""

    def test_basic_extraction(self):
        """Extract features from a minimal frame."""
        frame = {
            "contact": 1.0,
            "force_magnitude": 2.5,
            "force_direction": 0.3,
            "force_peak": 3.1,
            "deformation_magnitude": 0.1,
            "temporal_deformation_rate": 0.05,
            "slip_entropy": 0.8,
            "slip_event": 1.0,
            "texture_energy": 0.2,
            "contact_area": 100.0,
            "centroid_x": 50.0,
            "centroid_y": 75.0,
            "normal_mag": 1.5,
            "normal_var": 0.1,
            "shear_mag": 0.3,
            "shear_dir": 1.2,
            "delta_normal": 0.2,
            "delta_shear": 0.1,
            "friction_cone_ratio": 0.7,
        }
        result = extract_tactile_features(frame, "gelsight")

        assert result["observation.tactile.contact"] == [1.0]
        assert result["observation.tactile.force"] == [2.5, 0.3, 3.1]
        assert result["observation.tactile.deformation"] == [0.1, 0.05]
        assert result["observation.tactile.slip"] == [0.8, 1.0]
        assert result["observation.tactile.texture"] == [0.2]
        assert result["observation.tactile.contact_geometry"] == [100.0, 50.0, 75.0]
        assert result["observation.tactile.field"] == [1.5, 0.1, 0.3, 1.2]
        assert result["observation.tactile.dynamics"] == [0.2, 0.1, 0.7]

    def test_missing_values_default_to_zero(self):
        """Missing fields default to 0.0."""
        frame = {"contact": 1.0}
        result = extract_tactile_features(frame, "paxini")

        assert result["observation.tactile.contact"] == [1.0]
        assert result["observation.tactile.force"] == [0.0, 0.0, 0.0]
        assert result["observation.tactile.deformation"] == [0.0, 0.0]

    def test_empty_frame(self):
        """Empty frame returns all zeros."""
        result = extract_tactile_features({}, "gelsight")
        for key, value in result.items():
            assert all(v == 0.0 for v in value), f"{key} has non-zero values: {value}"


# ──────────────────────────────────────────────────────────────────────
# Sensor config tests (no tlabel/lerobot dependency)
# ──────────────────────────────────────────────────────────────────────
class TestSensorConfigs:
    """Test sensor-specific configurations."""

    def test_known_sensors(self):
        """All documented sensors are in config."""
        expected = {"gelsight", "digit", "paxini", "daimon", "touchd", "univtac", "vtac"}
        assert set(SENSOR_CONFIGS.keys()) == expected

    def test_visual_sensors_have_images(self):
        """Visual tactile sensors have image capability."""
        assert SENSOR_CONFIGS["gelsight"]["has_image"] is True
        assert SENSOR_CONFIGS["digit"]["has_image"] is True

    def test_force_sensors_no_images(self):
        """Force-based sensors have no image capability."""
        assert SENSOR_CONFIGS["paxini"]["has_image"] is False
        assert SENSOR_CONFIGS["daimon"]["has_image"] is False


# ──────────────────────────────────────────────────────────────────────
# Data loading tests (no tlabel/lerobot dependency)
# ──────────────────────────────────────────────────────────────────────
class TestDataLoading:
    """Test manual data loading fallback."""

    def test_load_json(self, tmp_path):
        """Load from tlabel_export.json."""
        data = {
            "metadata": {"sensor": "gelsight", "fps": 30},
            "frames": [
                {"episode_index": 0, "contact": 1.0, "force_magnitude": 2.0},
                {"episode_index": 0, "contact": 0.0, "force_magnitude": 0.0},
                {"episode_index": 1, "contact": 1.0, "force_magnitude": 1.5},
            ],
        }
        json_file = tmp_path / "tlabel_export.json"
        json_file.write_text(json.dumps(data))

        result = _load_manual(tmp_path)
        assert len(result["episodes"]) == 2
        assert len(result["episodes"][0]) == 2
        assert len(result["episodes"][1]) == 1

    def test_load_csv(self, tmp_path):
        """Load from CSV files."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "episode_index,contact,force_magnitude\n"
            "0,1.0,2.5\n"
            "0,0.0,0.0\n"
            "1,1.0,1.0\n"
        )

        result = _load_csv_episodes([csv_file])
        assert len(result["episodes"]) == 2
        assert result["episodes"][0][0]["contact"] == 1.0

    def test_no_data_raises(self, tmp_path):
        """Missing data files raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _load_manual(tmp_path)


# ──────────────────────────────────────────────────────────────────────
# Image dimension detection tests (no tlabel/lerobot dependency)
# ──────────────────────────────────────────────────────────────────────
class TestImageDimensionDetection:
    """Test dynamic image dimension detection."""

    def test_detect_from_numpy_array(self, tmp_path):
        """Detect dimensions from numpy array images."""
        episodes = {
            0: [{"tactile_image": np.zeros((240, 320, 3), dtype=np.uint8)}]
        }
        result = detect_image_dimensions(tmp_path, episodes)
        assert result == (240, 320)

    def test_detect_from_file(self, tmp_path):
        """Detect dimensions from image file path."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        # Create a test image
        img_array = np.zeros((100, 200, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = tmp_path / "test_image.png"
        img.save(img_path)

        episodes = {0: [{"tactile_image": "test_image.png"}]}
        result = detect_image_dimensions(tmp_path, episodes)
        assert result == (100, 200)

    def test_no_images_returns_none(self, tmp_path):
        """Return None when no images found."""
        episodes = {0: [{"contact": 1.0}]}
        result = detect_image_dimensions(tmp_path, episodes)
        assert result is None

    def test_empty_episodes_returns_none(self, tmp_path):
        """Return None for empty episodes."""
        result = detect_image_dimensions(tmp_path, {})
        assert result is None


# ──────────────────────────────────────────────────────────────────────
# End-to-end round-trip test (requires lerobot; tlabel mocked)
# ──────────────────────────────────────────────────────────────────────
class TestRoundTrip:
    """End-to-end round-trip: create dataset, add frames, finalize, reload."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample TLabel JSON data for testing."""
        data = {
            "metadata": {"sensor": "paxini", "fps": 30},
            "frames": [
                {
                    "episode_index": 0,
                    "contact": 1.0,
                    "force_magnitude": 2.5,
                    "force_direction": 0.3,
                    "force_peak": 3.1,
                    "deformation_magnitude": 0.1,
                    "temporal_deformation_rate": 0.05,
                    "slip_entropy": 0.8,
                    "slip_event": 1.0,
                    "texture_energy": 0.2,
                    "contact_area": 100.0,
                    "centroid_x": 50.0,
                    "centroid_y": 75.0,
                    "normal_mag": 1.5,
                    "normal_var": 0.1,
                    "shear_mag": 0.3,
                    "shear_dir": 1.2,
                    "delta_normal": 0.2,
                    "delta_shear": 0.1,
                    "friction_cone_ratio": 0.7,
                },
                {
                    "episode_index": 0,
                    "contact": 0.0,
                    "force_magnitude": 0.0,
                },
            ],
        }
        json_file = tmp_path / "tlabel_export.json"
        json_file.write_text(json.dumps(data))
        return tmp_path

    def test_roundtrip_no_image(self, sample_data, tmp_path):
        """Round-trip test without images (paxini sensor)."""
        lerobot = pytest.importorskip("lerobot")
        from lerobot.datasets import LeRobotDataset

        from convert_tlabel_to_lerobot import convert

        output_dir = str(tmp_path / "output")
        convert(
            input_dir=sample_data,
            repo_id="test/tactile_roundtrip",
            fps=30,
            sensor_type="paxini",
            output_dir=output_dir,
            task="test task",
        )

        # Reload and verify
        dataset = LeRobotDataset(
            "test/tactile_roundtrip",
            root=output_dir,
        )
        dataset.finalize()  # ensure finalized for reading

        assert len(dataset) == 2
        assert dataset.fps == 30
        assert "observation.tactile.contact" in dataset.features

        # Verify first frame data
        frame0 = dataset[0]
        assert frame0["observation.tactile.contact"].item() == pytest.approx(1.0)
        assert frame0["observation.tactile.force"][0].item() == pytest.approx(2.5)

    def test_roundtrip_with_numpy_images(self, sample_data, tmp_path):
        """Round-trip test with numpy array images (gelsight sensor)."""
        lerobot = pytest.importorskip("lerobot")
        from lerobot.datasets import LeRobotDataset

        # Add image data to sample
        json_file = sample_data / "tlabel_export.json"
        data = json.loads(json_file.read_text())

        # Replace with image data
        for frame in data["frames"]:
            frame["tactile_image"] = np.zeros((120, 160, 3), dtype=np.uint8)

        # Need to use direct construction since images are numpy arrays
        # Write as numpy references in a custom way
        from convert_tlabel_to_lerobot import (
            build_features,
            extract_tactile_features,
            _load_manual,
        )

        output_dir = str(tmp_path / "output_img")

        # Load data manually
        json_file.write_text(json.dumps({
            "metadata": {"sensor": "gelsight", "fps": 30},
            "frames": [
                {"episode_index": 0, "contact": 1.0, "force_magnitude": 2.0},
                {"episode_index": 0, "contact": 0.5, "force_magnitude": 1.0},
            ],
        }))

        manual_data = _load_manual(sample_data)
        episodes = manual_data["episodes"]

        # Add numpy images
        for ep_frames in episodes.values():
            for f in ep_frames:
                f["tactile_image"] = np.zeros((120, 160, 3), dtype=np.uint8)

        image_shape = detect_image_dimensions(sample_data, episodes)
        assert image_shape == (120, 160)

        features = build_features("gelsight", has_image=True, image_shape=image_shape)
        assert features["observation.images.tactile"]["shape"] == [120, 160, 3]

        dataset = LeRobotDataset.create(
            repo_id="test/tactile_img_roundtrip",
            fps=30,
            features=features,
            robot_type="gelsight",
            root=output_dir,
            use_videos=True,
        )

        for ep_idx, frames in sorted(episodes.items()):
            for frame_data in frames:
                frame = {}
                tactile = extract_tactile_features(frame_data, "gelsight")
                frame.update(tactile)
                frame["task"] = "test"
                frame["observation.images.tactile"] = frame_data["tactile_image"]
                dataset.add_frame(frame)
            dataset.save_episode(task="test")

        dataset.finalize()

        # Reload and verify
        reloaded = LeRobotDataset("test/tactile_img_roundtrip", root=output_dir)
        assert len(reloaded) == 2
        assert "observation.images.tactile" in reloaded.features
