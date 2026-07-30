"""Tests for TLabel to LeRobot conversion."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "tlabel"))

from convert_tlabel_to_lerobot import (
    SENSOR_CONFIGS,
    _load_csv_episodes,
    _load_manual,
    build_features,
    detect_image_dimensions,
    extract_tactile_features,
    validate_image_consistency,
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

        assert result["observation.tactile.contact"].tolist() == pytest.approx([1.0])
        assert result["observation.tactile.force"].tolist() == pytest.approx(
            [2.5, 0.3, 3.1]
        )
        assert result["observation.tactile.deformation"].tolist() == pytest.approx(
            [0.1, 0.05]
        )
        assert result["observation.tactile.slip"].tolist() == pytest.approx(
            [0.8, 1.0]
        )
        assert result["observation.tactile.texture"].tolist() == pytest.approx([0.2])
        assert result["observation.tactile.contact_geometry"].tolist() == pytest.approx(
            [100.0, 50.0, 75.0]
        )
        assert result["observation.tactile.field"].tolist() == pytest.approx(
            [1.5, 0.1, 0.3, 1.2]
        )
        assert result["observation.tactile.dynamics"].tolist() == pytest.approx(
            [0.2, 0.1, 0.7]
        )

    def test_returns_numpy_arrays(self):
        """All feature values are np.ndarray with float32 dtype."""
        result = extract_tactile_features({"contact": 1.0}, "paxini")
        for key, value in result.items():
            assert isinstance(value, np.ndarray), f"{key} is not np.ndarray: {type(value)}"
            assert value.dtype == np.float32, f"{key} dtype is {value.dtype}, expected float32"

    def test_missing_values_default_to_zero(self):
        """Missing fields default to 0.0."""
        frame = {"contact": 1.0}
        result = extract_tactile_features(frame, "paxini")

        assert result["observation.tactile.contact"].tolist() == pytest.approx([1.0])
        assert result["observation.tactile.force"].tolist() == pytest.approx(
            [0.0, 0.0, 0.0]
        )
        assert result["observation.tactile.deformation"].tolist() == pytest.approx(
            [0.0, 0.0]
        )

    def test_empty_frame(self):
        """Empty frame returns all zeros."""
        result = extract_tactile_features({}, "gelsight")
        for key, value in result.items():
            assert np.allclose(value, 0.0), f"{key} has non-zero values: {value}"


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
        episodes = {0: [{"tactile_image": np.zeros((240, 320, 3), dtype=np.uint8)}]}
        result = detect_image_dimensions(tmp_path, episodes)
        assert result == (240, 320)

    def test_detect_from_file(self, tmp_path):
        """Detect dimensions from image file path."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

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
# Image consistency validation tests (no tlabel/lerobot dependency)
# ──────────────────────────────────────────────────────────────────────
class TestImageConsistencyValidation:
    """Test that all frames are validated, not just the first one."""

    def test_all_frames_consistent_numpy(self, tmp_path):
        """All numpy frames with same shape pass validation."""
        episodes = {
            0: [
                {"tactile_image": np.zeros((100, 200, 3), dtype=np.uint8)},
                {"tactile_image": np.zeros((100, 200, 3), dtype=np.uint8)},
            ],
            1: [
                {"tactile_image": np.zeros((100, 200, 3), dtype=np.uint8)},
            ],
        }
        issues = validate_image_consistency(tmp_path, episodes, (100, 200))
        assert issues == []

    def test_later_frame_wrong_shape_numpy(self, tmp_path):
        """Detect resolution mismatch in a later frame (not just first)."""
        episodes = {
            0: [
                {"tactile_image": np.zeros((100, 200, 3), dtype=np.uint8)},
                {"tactile_image": np.zeros((50, 80, 3), dtype=np.uint8)},
            ],
        }
        issues = validate_image_consistency(tmp_path, episodes, (100, 200))
        assert len(issues) == 1
        assert "50, 80" in issues[0]

    def test_grayscale_numpy_detected(self, tmp_path):
        """Detect grayscale (2D) numpy arrays."""
        episodes = {
            0: [
                {"tactile_image": np.zeros((100, 200, 3), dtype=np.uint8)},
                {"tactile_image": np.zeros((100, 200), dtype=np.uint8)},
            ],
        }
        issues = validate_image_consistency(tmp_path, episodes, (100, 200))
        assert len(issues) == 1
        assert "1 channel" in issues[0]

    def test_rgba_numpy_detected(self, tmp_path):
        """Detect RGBA (4-channel) numpy arrays."""
        episodes = {
            0: [
                {"tactile_image": np.zeros((100, 200, 3), dtype=np.uint8)},
                {"tactile_image": np.zeros((100, 200, 4), dtype=np.uint8)},
            ],
        }
        issues = validate_image_consistency(tmp_path, episodes, (100, 200))
        assert len(issues) == 1
        assert "4 channel" in issues[0]

    def test_file_backed_consistency(self, tmp_path):
        """Validate file-backed images for consistency."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        for name in ["frame_000.png", "frame_001.png"]:
            img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
            img.save(tmp_path / name)

        bad_img = Image.fromarray(np.zeros((50, 80, 3), dtype=np.uint8))
        bad_img.save(tmp_path / "frame_002.png")

        episodes = {
            0: [
                {"tactile_image": "frame_000.png"},
                {"tactile_image": "frame_001.png"},
                {"tactile_image": "frame_002.png"},
            ],
        }
        issues = validate_image_consistency(tmp_path, episodes, (100, 200))
        assert len(issues) == 1
        assert "frame_002.png" in issues[0]

    def test_missing_file_detected(self, tmp_path):
        """Detect missing image file in later frames."""
        episodes = {
            0: [
                {"tactile_image": np.zeros((100, 200, 3), dtype=np.uint8)},
                {"tactile_image": "nonexistent.png"},
            ],
        }
        issues = validate_image_consistency(tmp_path, episodes, (100, 200))
        assert len(issues) == 1
        assert "not found" in issues[0]

    def test_no_image_frames_no_issues(self, tmp_path):
        """Frames without tactile_image key are skipped by validate_image_consistency."""
        episodes = {
            0: [{"contact": 1.0}, {"contact": 0.0}],
        }
        issues = validate_image_consistency(tmp_path, episodes, (100, 200))
        assert issues == []

    def test_none_tactile_image_skipped(self, tmp_path):
        """Frames with tactile_image=None are skipped by validate_image_consistency.

        The convert() function performs its own pre-validation to reject missing
        images before this function is called.
        """
        episodes = {
            0: [
                {"tactile_image": np.zeros((100, 200, 3), dtype=np.uint8)},
                {"tactile_image": None},
            ],
        }
        issues = validate_image_consistency(tmp_path, episodes, (100, 200))
        assert issues == []


# ──────────────────────────────────────────────────────────────────────
# End-to-end round-trip test (requires lerobot)
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
        pytest.importorskip("lerobot")
        pytest.importorskip("av")
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

        dataset = LeRobotDataset("test/tactile_roundtrip", root=output_dir)
        dataset.finalize()

        assert len(dataset) == 2
        assert dataset.fps == 30
        assert "observation.tactile.contact" in dataset.features

        frame0 = dataset[0]
        assert frame0["observation.tactile.contact"].item() == pytest.approx(1.0)
        assert frame0["observation.tactile.force"][0].item() == pytest.approx(2.5)

    def test_roundtrip_with_numpy_images(self, sample_data, tmp_path):
        """Round-trip test with numpy array images (gelsight sensor)."""
        pytest.importorskip("lerobot")
        pytest.importorskip("av")
        from lerobot.datasets import LeRobotDataset

        from convert_tlabel_to_lerobot import (
            _load_manual,
            build_features,
            detect_image_dimensions,
            extract_tactile_features,
        )

        json_file = sample_data / "tlabel_export.json"
        json_file.write_text(
            json.dumps(
                {
                    "metadata": {"sensor": "gelsight", "fps": 30},
                    "frames": [
                        {"episode_index": 0, "contact": 1.0, "force_magnitude": 2.0},
                        {"episode_index": 0, "contact": 0.5, "force_magnitude": 1.0},
                    ],
                }
            )
        )

        manual_data = _load_manual(sample_data)
        episodes = manual_data["episodes"]

        for ep_frames in episodes.values():
            for f in ep_frames:
                f["tactile_image"] = np.zeros((120, 160, 3), dtype=np.uint8)

        image_shape = detect_image_dimensions(sample_data, episodes)
        assert image_shape == (120, 160)

        features = build_features("gelsight", has_image=True, image_shape=image_shape)
        assert features["observation.images.tactile"]["shape"] == [120, 160, 3]

        output_dir = str(tmp_path / "output_img")
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

        reloaded = LeRobotDataset("test/tactile_img_roundtrip", root=output_dir)
        assert len(reloaded) == 2
        assert "observation.images.tactile" in reloaded.features

    def test_roundtrip_file_images_through_convert(self, tmp_path):
        """End-to-end round-trip: file-backed images go through convert()."""
        pytest.importorskip("lerobot")
        pytest.importorskip("av")
        from lerobot.datasets import LeRobotDataset
        from PIL import Image

        from convert_tlabel_to_lerobot import convert

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(3):
            arr = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img_dir / f"frame_{i:03d}.png")

        data = {
            "metadata": {"sensor": "gelsight", "fps": 30},
            "frames": [
                {
                    "episode_index": 0,
                    "contact": 1.0,
                    "force_magnitude": 2.0,
                    "tactile_image": f"images/frame_{i:03d}.png",
                }
                for i in range(3)
            ],
        }
        json_file = tmp_path / "tlabel_export.json"
        json_file.write_text(json.dumps(data))

        output_dir = str(tmp_path / "output_file_rt")
        convert(
            input_dir=tmp_path,
            repo_id="test/tactile_file_roundtrip",
            fps=30,
            sensor_type="gelsight",
            output_dir=output_dir,
            task="file image round-trip test",
        )

        dataset = LeRobotDataset("test/tactile_file_roundtrip", root=output_dir)
        assert len(dataset) == 3
        assert "observation.images.tactile" in dataset.features
        assert dataset.fps == 30

        frame0 = dataset[0]
        assert frame0["observation.tactile.contact"].item() == pytest.approx(1.0)
        assert frame0["observation.tactile.force"][0].item() == pytest.approx(2.0)
